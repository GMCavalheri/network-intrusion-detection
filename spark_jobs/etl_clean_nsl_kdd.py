"""
Spark ETL step 1 (NSL-KDD): read the pre-split KDDTrain+/KDDTest+ files,
attach the canonical column names (the raw files have no header row), derive
the binary is_attack label and a coarse cross-dataset attack_category, and
write cleaned Parquet partitioned by split plus a data-quality report.

NSL-KDD's "messy data" story isn't dirty values (it's a curated academic
dataset - né KDD'99, redone specifically to fix KDD'99's ~78% duplicate-row
problem) so this step instead verifies that claim (should find ~0 exact
duplicates) and reports the dataset's real defining property: KDDTest+
deliberately contains attack types that never appear in KDDTrain+, which is
what makes it a test of generalization to novel attacks rather than
memorization.

Run with:
    spark-submit etl_clean_nsl_kdd.py
"""

import json
import os

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

from common import (
    NSL_KDD_ATTACK_CATEGORY, NSL_KDD_CLEANED_PATH, NSL_KDD_COLUMNS,
    NSL_KDD_DIFFICULTY_COL, NSL_KDD_DQ_REPORT_PATH, NSL_KDD_LABEL_COL,
    RAW_NSL_KDD_DIR, get_logger, get_spark,
)

logger = get_logger("etl_clean_nsl_kdd")

_CATEGORICAL = {"protocol_type", "service", "flag"}


def _build_schema() -> StructType:
    fields = []
    for name in NSL_KDD_COLUMNS:
        dtype = StringType() if name in _CATEGORICAL else DoubleType()
        fields.append(StructField(name, dtype))
    fields.append(StructField(NSL_KDD_LABEL_COL, StringType()))
    fields.append(StructField(NSL_KDD_DIFFICULTY_COL, IntegerType()))
    return StructType(fields)


def read_split(spark, path: str, split_name: str):
    """Reads one of KDDTrain+.txt / KDDTest+.txt (headerless CSV) and tags
    every row with which split it came from."""
    df = spark.read.schema(_build_schema()).csv(path)
    return df.withColumn("split", F.lit(split_name))


def clean(train_df, test_df):
    """Pure transformation (no file I/O), so it's unit-testable against tiny
    in-memory DataFrames: union, dedupe, and derive is_attack/attack_category."""
    union_df = train_df.unionByName(test_df)

    total_before_dedup = union_df.count()
    deduped = union_df.dropDuplicates([c for c in NSL_KDD_COLUMNS] + [NSL_KDD_LABEL_COL, "split"])
    duplicates_removed = total_before_dedup - deduped.count()

    category_map = F.create_map([F.lit(x) for pair in NSL_KDD_ATTACK_CATEGORY.items() for x in pair])
    cleaned = deduped.withColumn(
        "is_attack", F.when(F.col(NSL_KDD_LABEL_COL) == "normal", F.lit(0)).otherwise(F.lit(1))
    ).withColumn(
        "attack_category", F.coalesce(category_map.getItem(F.col(NSL_KDD_LABEL_COL)), F.lit("Other"))
    )
    return cleaned, duplicates_removed


def main():
    logger.info("Starting etl_clean_nsl_kdd")
    spark = get_spark("nids-etl-clean-nsl-kdd")

    train_path = os.path.join(RAW_NSL_KDD_DIR, "KDDTrain+.txt")
    test_path = os.path.join(RAW_NSL_KDD_DIR, "KDDTest+.txt")
    train_df = read_split(spark, train_path, "train")
    test_df = read_split(spark, test_path, "test")

    train_raw = train_df.count()
    test_raw = test_df.count()
    logger.info("Read KDDTrain+: %d rows, KDDTest+: %d rows", train_raw, test_raw)

    train_attack_types = {r[NSL_KDD_LABEL_COL] for r in train_df.select(NSL_KDD_LABEL_COL).distinct().collect()}
    test_attack_types = {r[NSL_KDD_LABEL_COL] for r in test_df.select(NSL_KDD_LABEL_COL).distinct().collect()}
    novel_in_test = sorted(test_attack_types - train_attack_types)
    logger.info("Attack types in test but never seen in train (%d): %s", len(novel_in_test), novel_in_test)

    unmapped = sorted((train_attack_types | test_attack_types) - set(NSL_KDD_ATTACK_CATEGORY) - {"normal"})
    if unmapped:
        logger.warning("Attack type(s) with no category mapping, falling back to 'Other': %s", unmapped)

    cleaned, duplicates_removed = clean(train_df, test_df)
    final_count = cleaned.count()
    logger.info("Deduplicated: %d rows -> %d rows (%d duplicates removed)", train_raw + test_raw, final_count, duplicates_removed)

    logger.info("Writing cleaned Parquet to %s", NSL_KDD_CLEANED_PATH)
    cleaned.coalesce(4).write.mode("overwrite").partitionBy("split").parquet(NSL_KDD_CLEANED_PATH)

    report = {
        "raw_row_counts": {"KDDTrain+": train_raw, "KDDTest+": test_raw},
        "total_raw_rows": train_raw + test_raw,
        "duplicates_removed": duplicates_removed,
        "final_cleaned_rows": final_count,
        "distinct_attack_types_train": len(train_attack_types),
        "distinct_attack_types_test": len(test_attack_types),
        "novel_attack_types_in_test_not_in_train": novel_in_test,
        "attack_types_with_no_category_mapping": unmapped,
    }
    os.makedirs(os.path.dirname(NSL_KDD_DQ_REPORT_PATH), exist_ok=True)
    with open(NSL_KDD_DQ_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Data quality report written to %s", NSL_KDD_DQ_REPORT_PATH)
    logger.info("Data quality report: %s", json.dumps(report))

    spark.stop()
    logger.info("etl_clean_nsl_kdd complete")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("etl_clean_nsl_kdd failed")
        raise
