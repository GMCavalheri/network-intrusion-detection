"""
Spark ETL step 4: scores every row of both datasets with their respective
trained PipelineModel, maps each onto a common serving schema, and loads the
results into one shared Postgres table (`flows_scored`) for the FastAPI/
Streamlit layer to serve.

NSL-KDD and CICIDS2017 don't share raw feature columns (see common.py), so
this step maps each dataset's handful of dataset-agnostic fields (duration,
bytes sent/received, protocol where available) onto common columns for
cross-dataset queries/charts, and stores each row's *full* dataset-specific
feature vector as a JSONB `raw_features` column - the same "normalize the
common bits, keep the raw event as JSON" pattern real security data
pipelines use for heterogeneous log sources.

Run with:
    spark-submit --packages org.postgresql:postgresql:42.7.3 score_and_load.py
"""

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from common import (
    CICIDS2017_FEATURES_PATH, CICIDS2017_MODEL_PATH, CICIDS2017_NUMERIC_FEATURES,
    NSL_KDD_CATEGORICAL_FEATURES, NSL_KDD_FEATURES_PATH, NSL_KDD_MODEL_PATH,
    NSL_KDD_NUMERIC_FEATURES, get_logger, get_spark, postgres_config,
)

logger = get_logger("score_and_load")

COMMON_COLUMNS = [
    "flow_id", "dataset_source", "split", "source_day", "protocol",
    "duration", "bytes_sent", "bytes_received", "is_attack_actual",
    "attack_category_actual", "predicted_label", "attack_probability",
    "rule_flags", "raw_features",
]


def score_dataset(spark, features_path: str, model_path: str, dataset_source: str, raw_feature_cols: list[str]):
    """Loads a trained PipelineModel, scores its dataset's full feature
    table, and projects the result onto COMMON_COLUMNS. Kept separate from
    the Postgres write so it's testable in isolation given a pre-loaded
    PipelineModel and DataFrame (see tests/test_score_and_load.py)."""
    df = spark.read.parquet(features_path)
    model = PipelineModel.load(model_path)
    scored = model.transform(df).withColumn(
        "attack_probability", F.round(vector_to_array(F.col("probability"))[1], 5)
    )
    return project_to_common_schema(scored, dataset_source, raw_feature_cols)


def project_to_common_schema(scored_df, dataset_source: str, raw_feature_cols: list[str]):
    """Pure transformation (no Spark session/model needed beyond the already-
    scored DataFrame), so it's unit-testable against a tiny in-memory frame."""
    has_protocol = "protocol_type" in scored_df.columns
    has_source_day = "source_day" in scored_df.columns

    return scored_df.select(
        F.concat(F.lit(f"{dataset_source}-"), F.monotonically_increasing_id().cast("string")).alias("flow_id"),
        F.lit(dataset_source).alias("dataset_source"),
        F.col("split"),
        (F.col("source_day") if has_source_day else F.lit(None).cast("string")).alias("source_day"),
        (F.col("protocol_type") if has_protocol else F.lit(None).cast("string")).alias("protocol"),
        F.col("duration" if "duration" in scored_df.columns else "flow_duration").cast("double").alias("duration"),
        F.col("src_bytes" if "src_bytes" in scored_df.columns else "total_length_of_fwd_packets")
        .cast("double").alias("bytes_sent"),
        F.col("dst_bytes" if "dst_bytes" in scored_df.columns else "total_length_of_bwd_packets")
        .cast("double").alias("bytes_received"),
        F.col("is_attack").alias("is_attack_actual"),
        F.col("attack_category").alias("attack_category_actual"),
        F.col("prediction").cast("int").alias("predicted_label"),
        F.col("attack_probability"),
        F.col("rule_flags"),
        F.to_json(F.struct(*[F.col(c) for c in raw_feature_cols])).alias("raw_features"),
    )


def main():
    logger.info("Starting score_and_load")
    spark = get_spark("nids-score-and-load")

    nsl_kdd_scored = score_dataset(
        spark, NSL_KDD_FEATURES_PATH, NSL_KDD_MODEL_PATH, "nsl_kdd",
        NSL_KDD_NUMERIC_FEATURES + NSL_KDD_CATEGORICAL_FEATURES,
    )
    logger.info("Scored NSL-KDD: %d rows", nsl_kdd_scored.count())

    cicids2017_scored = score_dataset(
        spark, CICIDS2017_FEATURES_PATH, CICIDS2017_MODEL_PATH, "cicids2017",
        CICIDS2017_NUMERIC_FEATURES,
    )
    logger.info("Scored CICIDS2017: %d rows", cicids2017_scored.count())

    combined = nsl_kdd_scored.unionByName(cicids2017_scored)

    url, props = postgres_config()
    props_jsonb = {**props, "stringtype": "unspecified"}  # lets the driver cast raw_features' TEXT param into the jsonb column

    (
        combined.write.mode("overwrite")
        .option("truncate", "true")
        .jdbc(url, "flows_scored", properties=props_jsonb)
    )
    logger.info("Loaded %d scored flows into flows_scored", combined.count())

    dataset_stats = (
        combined.groupBy("dataset_source", "split")
        .agg(
            F.count("*").alias("total_flows"),
            F.sum("is_attack_actual").alias("actual_attack_count"),
            F.sum("predicted_label").alias("predicted_attack_count"),
            F.avg("attack_probability").alias("avg_attack_probability"),
        )
        .orderBy("dataset_source", "split")
    )
    (
        dataset_stats.write.mode("overwrite")
        .option("truncate", "true")
        .jdbc(url, "dataset_stats", properties=props)
    )
    logger.info("Loaded %d rows into dataset_stats", dataset_stats.count())

    spark.stop()
    logger.info("score_and_load complete")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("score_and_load failed")
        raise
