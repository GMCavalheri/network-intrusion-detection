"""
Spark ETL step 2 (NSL-KDD): derive interpretable rule-based flags from the
already-cleaned connection records.

Unlike the fraud-detection-spark reference project, this step does *not*
compute window-function velocity/history features: NSL-KDD rows are
independent, pre-aggregated connection records with no account/session
identifier and no timestamp to join across rows, so there's nothing to
window over. NSL-KDD's 41 columns are themselves already the product of
someone else's feature engineering (KDD'99's original per-connection and
per-2-second-window aggregates like `count`/`srv_count`/`*_rate`) - this step
just re-expresses a handful of them as human-readable rule flags, the same
role fraud's rule flags play: an interpretable baseline shown alongside the
trained model, not a replacement for it.

Run with:
    spark-submit feature_engineering_nsl_kdd.py
"""

from pyspark.sql import functions as F

from common import NSL_KDD_CLEANED_PATH, NSL_KDD_FEATURES_PATH, get_logger, get_spark

logger = get_logger("feature_engineering_nsl_kdd")


def engineer_features(df):
    """Pure transformation: cleaned NSL-KDD rows in, rule-flagged rows out.
    Extracted from main() so it's unit-testable against a tiny in-memory
    DataFrame without touching NSL_KDD_CLEANED_PATH/NSL_KDD_FEATURES_PATH."""
    df = df.withColumn(
        "rule_flag_failed_logins", F.when(F.col("num_failed_logins") >= 3, F.lit(1)).otherwise(F.lit(0))
    )
    df = df.withColumn("rule_flag_land_attack", F.when(F.col("land") == 1, F.lit(1)).otherwise(F.lit(0)))
    df = df.withColumn(
        "rule_flag_root_privilege",
        F.when((F.col("root_shell") == 1) | (F.col("num_root") > 0), F.lit(1)).otherwise(F.lit(0)),
    )
    # `count`/`srv_count` are already KDD's own 2-second-window connection
    # counts to the same host/service - a high value here is the direct NSL-
    # KDD analog of fraud's "high velocity" flag, no window function needed.
    df = df.withColumn(
        "rule_flag_high_connection_rate", F.when(F.col("count") >= 100, F.lit(1)).otherwise(F.lit(0))
    )
    df = df.withColumn(
        "rule_flag_high_error_rate",
        F.when((F.col("serror_rate") >= 0.5) | (F.col("rerror_rate") >= 0.5), F.lit(1)).otherwise(F.lit(0)),
    )

    df = df.withColumn(
        "rule_flags",
        F.array_join(
            F.filter(
                F.array(
                    F.when(F.col("rule_flag_failed_logins") == 1, F.lit("failed_logins")),
                    F.when(F.col("rule_flag_land_attack") == 1, F.lit("land_attack")),
                    F.when(F.col("rule_flag_root_privilege") == 1, F.lit("root_privilege")),
                    F.when(F.col("rule_flag_high_connection_rate") == 1, F.lit("high_connection_rate")),
                    F.when(F.col("rule_flag_high_error_rate") == 1, F.lit("high_error_rate")),
                ),
                lambda x: x.isNotNull(),
            ),
            ",",
        ),
    )
    return df


def main():
    logger.info("Starting feature_engineering_nsl_kdd")
    spark = get_spark("nids-feature-engineering-nsl-kdd")
    df = spark.read.parquet(NSL_KDD_CLEANED_PATH)
    logger.info("Read cleaned Parquet from %s", NSL_KDD_CLEANED_PATH)

    df = engineer_features(df)

    df = df.coalesce(4)
    logger.info("Writing feature Parquet to %s", NSL_KDD_FEATURES_PATH)
    df.write.mode("overwrite").partitionBy("split").parquet(NSL_KDD_FEATURES_PATH)
    logger.info("feature_engineering_nsl_kdd complete. Rows written: %d", df.count())

    spark.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("feature_engineering_nsl_kdd failed")
        raise
