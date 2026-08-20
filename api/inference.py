"""
Live scoring for the /score endpoint. Loads the exact PipelineModel trained
by spark_jobs/train_model.py for the requested dataset (via a lightweight
local SparkSession) so the API scores with the same model the cluster
produced, rather than a re-implemented copy - same reasoning as the
fraud-detection-spark reference project's api/inference.py.

NSL-KDD and CICIDS2017 need two independent models (different feature
spaces - see spark_jobs/common.py), so this module keeps a small cache of
{dataset: PipelineModel} and dispatches on the request's `dataset` field.
Rule flags are re-derived in plain Python from the same logic as
spark_jobs/feature_engineering_{nsl_kdd,cicids2017}.py, since the trained
models expect those rule_flag_* columns as part of their feature vector.
"""

import logging
import os
import time

from constants import (
    CICIDS2017_DEFAULTS, CICIDS2017_NUMERIC_FEATURES, DATASETS,
    NSL_KDD_CATEGORICAL_DEFAULTS, NSL_KDD_CATEGORICAL_FEATURES, NSL_KDD_DEFAULTS,
    NSL_KDD_NUMERIC_FEATURES,
)

logger = logging.getLogger("api")

S3_BUCKET = os.environ.get("MINIO_BUCKET", "network-intrusion-detection")
MODEL_PATHS = {
    "nsl_kdd": os.environ.get("NSL_KDD_MODEL_PATH", f"s3a://{S3_BUCKET}/models/nsl_kdd_model"),
    "cicids2017": os.environ.get("CICIDS2017_MODEL_PATH", f"s3a://{S3_BUCKET}/models/cicids2017_model"),
}

# Small, illustrative set of ports historically associated with malware C2 -
# duplicated from spark_jobs/feature_engineering_cicids2017.py on purpose
# (see constants.py's module docstring for why the API duplicates this data).
CICIDS2017_SUSPICIOUS_PORTS = [4444, 31337, 1337, 6667]

_spark = None
_models: dict = {}


def _get_spark():
    global _spark
    if _spark is None:
        from pyspark.sql import SparkSession

        _spark = (
            SparkSession.builder.appName("nids-api-inference")
            .master("local[2]")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.ui.enabled", "false")
            .config(
                "spark.jars.packages",
                "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262",
            )
            .config("spark.hadoop.fs.s3a.endpoint", os.environ.get("MINIO_ENDPOINT", "http://minio:9000"))
            .config("spark.hadoop.fs.s3a.access.key", os.environ.get("MINIO_ACCESS_KEY", "minioadmin"))
            .config("spark.hadoop.fs.s3a.secret.key", os.environ.get("MINIO_SECRET_KEY", "minioadmin"))
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
            .getOrCreate()
        )
    return _spark


def _get_model(dataset: str):
    if dataset not in _models:
        from pyspark.ml import PipelineModel

        logger.info("Loading %s model from %s (first call for this dataset - takes a few seconds)", dataset, MODEL_PATHS[dataset])
        start = time.monotonic()
        spark = _get_spark()
        _models[dataset] = PipelineModel.load(MODEL_PATHS[dataset])
        logger.info("%s model loaded in %.1fs", dataset, time.monotonic() - start)
    return _models[dataset]


def _rule_flags_nsl_kdd(f: dict) -> list[str]:
    flags = []
    if f["num_failed_logins"] >= 3:
        flags.append("failed_logins")
    if f["land"] == 1:
        flags.append("land_attack")
    if f["root_shell"] == 1 or f["num_root"] > 0:
        flags.append("root_privilege")
    if f["count"] >= 100:
        flags.append("high_connection_rate")
    if f["serror_rate"] >= 0.5 or f["rerror_rate"] >= 0.5:
        flags.append("high_error_rate")
    return flags


def _rule_flags_cicids2017(f: dict) -> list[str]:
    flags = []
    if f["total_length_of_fwd_packets"] == 0 and f["total_length_of_bwd_packets"] == 0:
        flags.append("zero_payload")
    if f["syn_flag_count"] >= 1 and f["ack_flag_count"] == 0:
        flags.append("syn_flood_like")
    if f["destination_port"] in CICIDS2017_SUSPICIOUS_PORTS:
        flags.append("malicious_port")
    if f["total_backward_packets"] == 0:
        flags.append("asymmetric_flow")
    return flags


def build_features(dataset: str, provided: dict) -> tuple[dict, list[str]]:
    """Merges the user-supplied subset of fields with sensible defaults for
    the rest (see constants.py's *_DEFAULTS), then derives the same
    rule_flag_* columns the trained model was fit with. Raises ValueError on
    an unknown dataset or an unrecognized feature key."""
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}, got {dataset!r}")

    if dataset == "nsl_kdd":
        allowed = set(NSL_KDD_NUMERIC_FEATURES) | set(NSL_KDD_CATEGORICAL_FEATURES)
        unknown = set(provided) - allowed
        if unknown:
            raise ValueError(f"Unknown NSL-KDD feature(s): {sorted(unknown)}")

        numeric = dict(NSL_KDD_DEFAULTS)
        categorical = dict(NSL_KDD_CATEGORICAL_DEFAULTS)
        for k, v in provided.items():
            if k in NSL_KDD_CATEGORICAL_FEATURES:
                categorical[k] = v
            else:
                numeric[k] = float(v)

        rule_flags = _rule_flags_nsl_kdd(numeric)
        features = {**numeric, **categorical}
        for flag_name in ("failed_logins", "land_attack", "root_privilege", "high_connection_rate", "high_error_rate"):
            features[f"rule_flag_{flag_name}"] = 1 if flag_name in rule_flags else 0
        return features, rule_flags

    # cicids2017
    allowed = set(CICIDS2017_NUMERIC_FEATURES)
    unknown = set(provided) - allowed
    if unknown:
        raise ValueError(f"Unknown CICIDS2017 feature(s): {sorted(unknown)}")

    numeric = dict(CICIDS2017_DEFAULTS)
    for k, v in provided.items():
        numeric[k] = float(v)

    rule_flags = _rule_flags_cicids2017(numeric)
    features = dict(numeric)
    for flag_name in ("zero_payload", "syn_flood_like", "malicious_port", "asymmetric_flow"):
        features[f"rule_flag_{flag_name}"] = 1 if flag_name in rule_flags else 0
    return features, rule_flags


def score(dataset: str, provided_features: dict) -> dict:
    features, rule_flags = build_features(dataset, provided_features)
    spark = _get_spark()
    model = _get_model(dataset)

    row_df = spark.createDataFrame([features])
    prediction = model.transform(row_df).select("prediction", "probability").first()
    attack_probability = float(prediction["probability"][1])
    predicted_label = int(prediction["prediction"])

    logger.info(
        "Scored dataset=%s -> probability=%.4f label=%d flags=%s",
        dataset, attack_probability, predicted_label, rule_flags,
    )

    return {
        "dataset": dataset,
        "attack_probability": round(attack_probability, 5),
        "predicted_label": predicted_label,
        "rule_flags": rule_flags,
        "features_used": features,
    }
