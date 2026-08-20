"""
Spark ETL step 2 (CICIDS2017): derive interpretable rule-based flags from the
already-cleaned flow records.

Like feature_engineering_nsl_kdd.py, this does *not* compute window-function
velocity/history features: this CICFlowMeter CSV release has no source/
destination IP column (stripped upstream, verified against the real file -
see common.CICIDS2017_RAW_COLUMNS), so there's no entity to group by for a
"connections from this host in the last minute" style feature the way the
fraud-detection-spark reference does with account_id. Each row already *is*
one fully-aggregated flow. This step instead flags flow-intrinsic patterns
that read as suspicious on their own - the interpretable-baseline role rule
flags play throughout this project.

Run with:
    spark-submit feature_engineering_cicids2017.py
"""

from pyspark.sql import functions as F

from common import CICIDS2017_CLEANED_PATH, CICIDS2017_FEATURES_PATH, get_logger, get_spark

logger = get_logger("feature_engineering_cicids2017")

# Small, illustrative set of ports historically associated with malware C2 /
# backdoors (Metasploit default, Back Orifice/"elite", common IRC botnet
# C2) - not exhaustive, a real deployment would pull a maintained threat-intel
# feed instead of a hardcoded list.
SUSPICIOUS_PORTS = [4444, 31337, 1337, 6667]


def engineer_features(df):
    """Pure transformation: cleaned CICIDS2017 rows in, rule-flagged rows
    out. Extracted from main() so it's unit-testable against a tiny
    in-memory DataFrame without touching CICIDS2017_CLEANED_PATH/FEATURES_PATH."""
    df = df.withColumn(
        "rule_flag_zero_payload",
        F.when(
            (F.col("total_length_of_fwd_packets") == 0) & (F.col("total_length_of_bwd_packets") == 0),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )
    df = df.withColumn(
        "rule_flag_syn_flood_like",
        F.when((F.col("syn_flag_count") >= 1) & (F.col("ack_flag_count") == 0), F.lit(1)).otherwise(F.lit(0)),
    )
    df = df.withColumn(
        "rule_flag_malicious_port",
        F.when(F.col("destination_port").isin(SUSPICIOUS_PORTS), F.lit(1)).otherwise(F.lit(0)),
    )
    df = df.withColumn(
        "rule_flag_asymmetric_flow",
        F.when(F.col("total_backward_packets") == 0, F.lit(1)).otherwise(F.lit(0)),
    )

    df = df.withColumn(
        "rule_flags",
        F.array_join(
            F.filter(
                F.array(
                    F.when(F.col("rule_flag_zero_payload") == 1, F.lit("zero_payload")),
                    F.when(F.col("rule_flag_syn_flood_like") == 1, F.lit("syn_flood_like")),
                    F.when(F.col("rule_flag_malicious_port") == 1, F.lit("malicious_port")),
                    F.when(F.col("rule_flag_asymmetric_flow") == 1, F.lit("asymmetric_flow")),
                ),
                lambda x: x.isNotNull(),
            ),
            ",",
        ),
    )
    return df


def main():
    logger.info("Starting feature_engineering_cicids2017")
    spark = get_spark("nids-feature-engineering-cicids2017")
    df = spark.read.parquet(CICIDS2017_CLEANED_PATH)
    logger.info("Read cleaned Parquet from %s", CICIDS2017_CLEANED_PATH)

    df = engineer_features(df)

    df = df.coalesce(4)
    logger.info("Writing feature Parquet to %s", CICIDS2017_FEATURES_PATH)
    df.write.mode("overwrite").partitionBy("split").parquet(CICIDS2017_FEATURES_PATH)
    logger.info("feature_engineering_cicids2017 complete. Rows written: %d", df.count())

    spark.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("feature_engineering_cicids2017 failed")
        raise
