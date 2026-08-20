"""
Spark ETL step 3: trains a GBT classifier for EACH dataset (NSL-KDD and
CICIDS2017 get their own independent PipelineModel - see common.py's module
docstring for why their feature spaces can't share one model) and writes a
fitted Spark MLlib PipelineModel + an evaluation report per dataset.

Both datasets already carry their own train/test split from etl_clean_*.py:
NSL-KDD uses its native KDDTrain+/KDDTest+ split (famous for containing
attack types in test that never appear in train - this script explicitly
reports the model's detection rate on exactly those novel-attack rows,
the real test of whether it generalizes rather than memorizes). CICIDS2017
uses a day-based split (Monday+Wednesday train, Friday test) since this CSV
release has no timestamp to split on within a day - see etl_clean_cicids2017.py.

Run with:
    spark-submit train_model.py
"""

import json
import os

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from common import (
    CICIDS2017_FEATURES_PATH, CICIDS2017_METRICS_PATH, CICIDS2017_MODEL_PATH,
    CICIDS2017_NUMERIC_FEATURES, NSL_KDD_CATEGORICAL_FEATURES,
    NSL_KDD_FEATURES_PATH, NSL_KDD_LABEL_COL, NSL_KDD_METRICS_PATH, NSL_KDD_MODEL_PATH,
    NSL_KDD_NUMERIC_FEATURES, get_logger, get_spark,
)

logger = get_logger("train_model")

LABEL_COL = "is_attack"
RULE_FLAG_PREFIX = "rule_flag_"


def _rule_flag_columns(df) -> list[str]:
    return sorted(c for c in df.columns if c.startswith(RULE_FLAG_PREFIX))


def _threshold_curve(predictions, resolution: float = 0.01) -> list[dict]:
    """Precision/recall/F1 across the full range of classification
    thresholds, computed with one aggregation query (bucket the predicted
    probability, then sweep cumulative counts from p=1 down to p=0) rather
    than one .count() per candidate threshold.

    Exists because the *default* 0.5 threshold can be badly miscalibrated
    for gradient-boosted trees on a real train/test distribution shift (e.g.
    CICIDS2017's day-based split: verified on the real 3-day dataset, the
    model trained on Monday/Wednesday scores Friday's actual DDoS flows at a
    median probability of ~0.04, not >0.5 - despite ranking them well above
    benign flows, giving a strong AUC-ROC. Reporting only the @0.5 metrics
    would make a genuinely well-ranking model look broken; a security team
    tuning an alert threshold to their false-positive budget needs this
    curve, not just one arbitrary operating point.
    """
    buckets = (
        predictions.withColumn("p_bucket", F.round(F.col("attack_probability") / resolution) * resolution)
        .groupBy("p_bucket")
        .agg(F.count("*").alias("n"), F.sum(F.col(LABEL_COL).cast("long")).alias("pos"))
        .collect()
    )
    buckets = sorted(buckets, key=lambda r: -r["p_bucket"])
    total_pos = sum(r["pos"] for r in buckets)

    curve = []
    cum_n = cum_pos = 0
    for r in buckets:
        cum_n += r["n"]
        cum_pos += r["pos"]
        precision = cum_pos / cum_n if cum_n else 0.0
        recall = cum_pos / total_pos if total_pos else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        curve.append({
            "threshold": round(float(r["p_bucket"]), 5),
            "precision": round(precision, 5), "recall": round(recall, 5), "f1": round(f1, 5),
        })
    return curve


def _compute_metrics(predictions, feature_cols, gbt_model, extra: dict) -> dict:
    evaluator_roc = BinaryClassificationEvaluator(labelCol=LABEL_COL, metricName="areaUnderROC")
    evaluator_pr = BinaryClassificationEvaluator(labelCol=LABEL_COL, metricName="areaUnderPR")
    auc_roc = evaluator_roc.evaluate(predictions)
    auc_pr = evaluator_pr.evaluate(predictions)

    counts = {
        (int(r[LABEL_COL]), int(r["prediction"])): r["count"]
        for r in predictions.groupBy(LABEL_COL, "prediction").count().collect()
    }
    tp = counts.get((1, 1), 0)
    fp = counts.get((0, 1), 0)
    tn = counts.get((0, 0), 0)
    fn = counts.get((1, 0), 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    importances = gbt_model.featureImportances.toArray()
    feature_importances = sorted(
        zip(feature_cols, [round(float(x), 5) for x in importances]), key=lambda x: -x[1],
    )

    category_breakdown = [
        {
            "attack_category": r["attack_category"],
            "count": r["count"],
            "predicted_attack_count": int(r["predicted_attack_count"]),
            "detection_rate": round(r["predicted_attack_count"] / r["count"], 5) if r["count"] else None,
        }
        for r in predictions.groupBy("attack_category")
        .agg(F.count("*").alias("count"), F.sum("prediction").alias("predicted_attack_count"))
        .orderBy("attack_category")
        .collect()
    ]

    threshold_curve = _threshold_curve(predictions)
    best_point = max(threshold_curve, key=lambda p: p["f1"]) if threshold_curve else None

    metrics = {
        "test_rows": predictions.count(),
        "attack_rate_test": round(predictions.filter(F.col(LABEL_COL) == 1).count() / predictions.count(), 5),
        "auc_roc": round(auc_roc, 5),
        "auc_pr": round(auc_pr, 5),
        # @0.5 (Spark's default decision threshold) - can look much worse
        # than auc_roc/auc_pr suggest if the model is miscalibrated on this
        # test set; see threshold_curve/best_threshold for the full picture.
        "precision": round(precision, 5),
        "recall": round(recall, 5),
        "f1": round(f1, 5),
        "confusion_matrix": {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn},
        "threshold_curve": threshold_curve,
        "best_threshold": best_point,
        "feature_importances": feature_importances,
        "feature_columns": feature_cols,
        "category_breakdown": category_breakdown,
        "model_type": "GBTClassifier",
    }
    metrics.update(extra)
    return metrics


def train_and_evaluate(train_df, test_df, numeric_features, categorical_features, extra_metrics=None, seed=42):
    """Fits a GBTClassifier pipeline and evaluates it on test_df. Pure
    function over already-loaded DataFrames (no file I/O), so it's
    unit-testable against tiny in-memory frames."""
    rule_flags = _rule_flag_columns(train_df)
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in categorical_features
    ]
    feature_cols = numeric_features + rule_flags + [f"{c}_idx" for c in categorical_features]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")

    train_pos = train_df.filter(F.col(LABEL_COL) == 1).count()
    train_neg = train_df.filter(F.col(LABEL_COL) == 0).count()
    weight_pos = (train_neg / train_pos) if train_pos else 1.0
    train_df = train_df.withColumn(
        "class_weight", F.when(F.col(LABEL_COL) == 1, F.lit(weight_pos)).otherwise(F.lit(1.0))
    )

    gbt = GBTClassifier(
        labelCol=LABEL_COL, featuresCol="features", weightCol="class_weight",
        # maxBins must be >= the largest categorical feature's cardinality or
        # GBTClassifier throws IllegalArgumentException at fit() time - NSL-
        # KDD's `service` column alone has ~70 distinct values (verified
        # against real data), so the Spark default of 32 isn't enough.
        maxIter=50, maxDepth=5, maxBins=128, seed=seed,
    )
    pipeline = Pipeline(stages=indexers + [assembler, gbt])
    model = pipeline.fit(train_df)

    predictions = model.transform(test_df).withColumn(
        "attack_probability", vector_to_array(F.col("probability"))[1]
    )
    predictions.cache()

    metrics = _compute_metrics(predictions, feature_cols, model.stages[-1], extra_metrics or {})
    metrics["train_rows"] = train_df.count()
    metrics["attack_rate_train"] = round(train_pos / (train_pos + train_neg), 5) if (train_pos + train_neg) else 0
    return model, metrics


def train_nsl_kdd(spark):
    logger.info("Training NSL-KDD model")
    df = spark.read.parquet(NSL_KDD_FEATURES_PATH)
    train_df = df.filter(F.col("split") == "train")
    test_df = df.filter(F.col("split") == "test")

    train_types = {r[NSL_KDD_LABEL_COL] for r in train_df.select(NSL_KDD_LABEL_COL).distinct().collect()}
    test_types = {r[NSL_KDD_LABEL_COL] for r in test_df.select(NSL_KDD_LABEL_COL).distinct().collect()}
    novel_types = sorted(test_types - train_types)

    model, metrics = train_and_evaluate(
        train_df, test_df, NSL_KDD_NUMERIC_FEATURES, NSL_KDD_CATEGORICAL_FEATURES,
        extra_metrics={"split_strategy": "native KDDTrain+/KDDTest+ split"},
    )

    # The headline metric for NSL-KDD: recall computed only over rows whose
    # attack_type never appeared during training - does the model generalize
    # to genuinely novel attacks, or did it just memorize the training set?
    if novel_types:
        novel_df = test_df.filter(F.col(NSL_KDD_LABEL_COL).isin(novel_types))
        novel_predictions = model.transform(novel_df)
        novel_total = novel_predictions.count()
        novel_caught = novel_predictions.filter(F.col("prediction") == 1).count()
        metrics["novel_attack_types_in_test"] = novel_types
        metrics["novel_attack_detection_rate"] = round(novel_caught / novel_total, 5) if novel_total else None
        logger.info(
            "Novel-attack detection rate (attack types never seen in training): %s (%d/%d rows)",
            metrics["novel_attack_detection_rate"], novel_caught, novel_total,
        )

    logger.info(
        "NSL-KDD evaluation: auc_roc=%.5f auc_pr=%.5f precision=%.5f recall=%.5f f1=%.5f",
        metrics["auc_roc"], metrics["auc_pr"], metrics["precision"], metrics["recall"], metrics["f1"],
    )
    os.makedirs(os.path.dirname(NSL_KDD_METRICS_PATH), exist_ok=True)
    with open(NSL_KDD_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("NSL-KDD metrics written to %s", NSL_KDD_METRICS_PATH)

    model.write().overwrite().save(NSL_KDD_MODEL_PATH)
    logger.info("NSL-KDD model saved to %s", NSL_KDD_MODEL_PATH)


def train_cicids2017(spark):
    logger.info("Training CICIDS2017 model")
    df = spark.read.parquet(CICIDS2017_FEATURES_PATH)
    train_df = df.filter(F.col("split") == "train")
    test_df = df.filter(F.col("split") == "test")

    model, metrics = train_and_evaluate(
        train_df, test_df, CICIDS2017_NUMERIC_FEATURES, [],
        extra_metrics={"split_strategy": "day-based split (train/test days in common.py)"},
    )

    logger.info(
        "CICIDS2017 evaluation: auc_roc=%.5f auc_pr=%.5f precision=%.5f recall=%.5f f1=%.5f",
        metrics["auc_roc"], metrics["auc_pr"], metrics["precision"], metrics["recall"], metrics["f1"],
    )
    os.makedirs(os.path.dirname(CICIDS2017_METRICS_PATH), exist_ok=True)
    with open(CICIDS2017_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("CICIDS2017 metrics written to %s", CICIDS2017_METRICS_PATH)

    model.write().overwrite().save(CICIDS2017_MODEL_PATH)
    logger.info("CICIDS2017 model saved to %s", CICIDS2017_MODEL_PATH)


def main():
    logger.info("Starting train_model")
    spark = get_spark("nids-model-training")

    train_nsl_kdd(spark)
    train_cicids2017(spark)

    spark.stop()
    logger.info("train_model complete")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("train_model failed")
        raise
