"""
Spark ETL step 1 (CICIDS2017): read the per-day CICFlowMeter CSVs, apply the
project's canonical (positional) column names, clean up the dataset's real,
well-documented data-quality issues, and write cleaned Parquet partitioned by
day plus a data-quality report.

Verified directly against a downloaded day's CSV (not just documentation):
  - every column name but 4 has a stray leading space, and column 55 ("Fwd
    Header Length") is a byte-for-byte duplicate of column 34 - see
    common.CICIDS2017_RAW_COLUMNS for why this reads the file positionally
    instead of by header name
  - "Flow Bytes/s" and "Flow Packets/s" are literal string "Infinity"/"NaN"
    when a flow's duration is 0 (division by zero in CICFlowMeter) - 30
    Infinity + 4 NaN values in one 225k-row day alone
  - a small number of rows (2 in the same day) have a negative Flow Duration,
    which is physically invalid and dropped rather than "corrected" (unlike
    a negative amount, there's no sign convention to flip)
  - ~1.2% of rows in that same day are byte-for-byte exact duplicates

Run with:
    spark-submit etl_clean_cicids2017.py
"""

import json
import os

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType, StructField, StructType

from common import (
    CICIDS2017_CLEANED_PATH, CICIDS2017_DAY_FILES, CICIDS2017_DQ_REPORT_PATH,
    CICIDS2017_LABEL_CATEGORY, CICIDS2017_LABEL_COL, CICIDS2017_RATE_COLUMNS,
    CICIDS2017_RAW_COLUMNS, CICIDS2017_TEST_DAYS, CICIDS2017_TRAIN_DAYS,
    RAW_CICIDS2017_DIR, get_logger, get_spark,
)

logger = get_logger("etl_clean_cicids2017")

_INT_COLUMNS = {"destination_port"}
_LONG_COLUMNS = {"flow_duration"}


def _build_schema() -> StructType:
    fields = []
    for name in CICIDS2017_RAW_COLUMNS:
        if name == "label":
            dtype = StringType()
        elif name in _INT_COLUMNS:
            dtype = IntegerType()
        elif name in _LONG_COLUMNS:
            dtype = LongType()
        else:
            dtype = DoubleType()  # Java's Double.parseDouble natively accepts "Infinity"/"NaN"
        fields.append(StructField(name, dtype))
    return StructType(fields)


def read_day(spark, path: str, day: str):
    """Reads one CICIDS2017 day CSV positionally (see module docstring for
    why: duplicate header names would make Spark reject the file if read by
    name) and tags every row with which capture day and train/test split it
    belongs to."""
    df = spark.read.option("header", True).schema(_build_schema()).csv(path)
    split = "train" if day in CICIDS2017_TRAIN_DAYS else "test"
    return df.drop("fwd_header_length_dup").withColumn("source_day", F.lit(day)).withColumn("split", F.lit(split))


def clean(df):
    """Pure transformation (no file I/O): drops physically-invalid rows,
    imputes Infinity/NaN rate values with a -1 sentinel, dedupes, and derives
    attack_category. Returns (cleaned_df, stats_dict) so it's unit-testable
    against a tiny in-memory DataFrame.

    Infinity/NaN rate values are imputed to -1.0, not left null: downstream,
    VectorAssembler(handleInvalid="keep") turns a null *numeric* input into
    NaN in the assembled vector (its "keep" handling is for indexer/encoder
    categories, not this), and GBTClassifier's tree-metadata step rejects any
    NaN in the feature vector outright - confirmed by actually running this
    against the real 3-day dataset, where training crashed on exactly the
    rows this step nulls out. -1 is an unambiguous "rate undefined because
    duration was 0" sentinel, same convention the fraud-detection-spark
    reference project uses for its own undefined-value features.
    """
    total_before = df.count()

    negative_duration = df.filter(F.col("flow_duration") < 0).count()
    df = df.filter(F.col("flow_duration") >= 0)

    rate_issue_counts = {}
    for c in CICIDS2017_RATE_COLUMNS:
        is_bad = F.isnan(F.col(c)) | (F.abs(F.col(c)) == float("inf"))
        rate_issue_counts[c] = df.filter(is_bad).count()
        df = df.withColumn(c, F.when(is_bad, F.lit(-1.0)).otherwise(F.col(c)))

    before_dedup = df.count()
    df = df.dropDuplicates([c for c in df.columns if c not in ("source_day", "split")])
    duplicates_removed = before_dedup - df.count()

    category_map = F.create_map([F.lit(x) for pair in CICIDS2017_LABEL_CATEGORY.items() for x in pair])
    df = df.withColumn(
        "is_attack", F.when(F.col(CICIDS2017_LABEL_COL) == "BENIGN", F.lit(0)).otherwise(F.lit(1))
    ).withColumn(
        "attack_category", F.coalesce(category_map.getItem(F.col(CICIDS2017_LABEL_COL)), F.lit("Other"))
    )

    stats = {
        "total_before_cleaning": total_before,
        "negative_duration_rows_dropped": negative_duration,
        "rate_column_infinity_or_nan_imputed": rate_issue_counts,
        "duplicates_removed": duplicates_removed,
        "final_cleaned_rows": df.count(),
    }
    return df, stats


def main():
    logger.info("Starting etl_clean_cicids2017")
    spark = get_spark("nids-etl-clean-cicids2017")

    days = CICIDS2017_TRAIN_DAYS + CICIDS2017_TEST_DAYS
    per_day_raw_counts = {}
    day_frames = []
    for day in days:
        path = os.path.join(RAW_CICIDS2017_DIR, CICIDS2017_DAY_FILES[day])
        day_df = read_day(spark, path, day)
        n = day_df.count()
        per_day_raw_counts[day] = n
        logger.info("Read %s (%s): %d rows", day, CICIDS2017_DAY_FILES[day], n)
        day_frames.append(day_df)

    union_df = day_frames[0]
    for f in day_frames[1:]:
        union_df = union_df.unionByName(f)

    label_values = {r[CICIDS2017_LABEL_COL] for r in union_df.select(CICIDS2017_LABEL_COL).distinct().collect()}
    unmapped = sorted(label_values - set(CICIDS2017_LABEL_CATEGORY))
    if unmapped:
        logger.warning("Label(s) with no category mapping, falling back to 'Other': %s", unmapped)

    cleaned, stats = clean(union_df)
    logger.info(
        "Cleaned: %d rows -> %d rows (%d duplicates removed, %d negative-duration rows dropped)",
        stats["total_before_cleaning"], stats["final_cleaned_rows"],
        stats["duplicates_removed"], stats["negative_duration_rows_dropped"],
    )

    logger.info("Writing cleaned Parquet to %s", CICIDS2017_CLEANED_PATH)
    cleaned.coalesce(4).write.mode("overwrite").partitionBy("split").parquet(CICIDS2017_CLEANED_PATH)

    report = {
        "days_included": days,
        "train_days": CICIDS2017_TRAIN_DAYS,
        "test_days": CICIDS2017_TEST_DAYS,
        "raw_row_counts_by_day": per_day_raw_counts,
        "total_raw_rows": sum(per_day_raw_counts.values()),
        **stats,
        "labels_with_no_category_mapping": unmapped,
    }
    os.makedirs(os.path.dirname(CICIDS2017_DQ_REPORT_PATH), exist_ok=True)
    with open(CICIDS2017_DQ_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Data quality report written to %s", CICIDS2017_DQ_REPORT_PATH)
    logger.info("Data quality report: %s", json.dumps(report))

    spark.stop()
    logger.info("etl_clean_cicids2017 complete")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("etl_clean_cicids2017 failed")
        raise
