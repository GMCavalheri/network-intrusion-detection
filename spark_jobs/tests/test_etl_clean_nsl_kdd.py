import os

import etl_clean_nsl_kdd as ec
from common import NSL_KDD_COLUMNS
from pyspark.sql import functions as F


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _default_row(**overrides):
    """A full 43-column NSL-KDD row (41 features + attack_type + difficulty)
    with every feature defaulted to 0.0/"" so tests only need to specify the
    handful of columns they actually care about."""
    row = {c: ("" if c in ("protocol_type", "service", "flag") else 0.0) for c in NSL_KDD_COLUMNS}
    row["attack_type"] = "normal"
    row["difficulty"] = 15
    row.update(overrides)
    return tuple(row[c] for c in NSL_KDD_COLUMNS) + (row["attack_type"], row["difficulty"])


def _make_df(spark, rows, split_name):
    df = spark.createDataFrame(list(rows), schema=ec._build_schema())
    return df.withColumn("split", F.lit(split_name))


class TestReadSplit:
    def test_parses_headerless_csv_with_canonical_columns(self, spark, tmp_path):
        path = os.path.join(str(tmp_path), "KDDTrain+.txt")
        _write(
            path,
            "0,tcp,ftp_data,SF,491,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,0.00,0.00,0.00,0.00,1.00,0.00,0.00,150,25,0.17,0.03,0.17,0.00,0.00,0.00,0.05,0.00,normal,20\n",
        )

        df = ec.read_split(spark, path, "train")
        row = df.collect()[0]

        assert row["protocol_type"] == "tcp"
        assert row["service"] == "ftp_data"
        assert row["flag"] == "SF"
        assert row["src_bytes"] == 491.0
        assert row["attack_type"] == "normal"
        assert row["difficulty"] == 20
        assert row["split"] == "train"


class TestClean:
    def test_derives_is_attack_binary_label(self, spark):
        train_df = _make_df(
            spark,
            [
                _default_row(attack_type="normal"),
                _default_row(attack_type="neptune"),
            ],
            "train",
        )
        test_df = _make_df(spark, [], "test")

        cleaned, _ = ec.clean(train_df, test_df)
        rows = {r["attack_type"]: r for r in cleaned.collect()}

        assert rows["normal"]["is_attack"] == 0
        assert rows["neptune"]["is_attack"] == 1

    def test_maps_known_attack_types_to_unified_category(self, spark):
        train_df = _make_df(
            spark,
            [
                _default_row(attack_type="neptune"),   # DoS
                _default_row(attack_type="satan"),      # Probe/Scan
                _default_row(attack_type="buffer_overflow"),  # Privilege Escalation
            ],
            "train",
        )
        test_df = _make_df(spark, [], "test")

        cleaned, _ = ec.clean(train_df, test_df)
        rows = {r["attack_type"]: r for r in cleaned.collect()}

        assert rows["neptune"]["attack_category"] == "DoS"
        assert rows["satan"]["attack_category"] == "Probe/Scan"
        assert rows["buffer_overflow"]["attack_category"] == "Privilege Escalation"

    def test_unmapped_attack_type_falls_back_to_other(self, spark):
        train_df = _make_df(spark, [_default_row(attack_type="totally_novel_attack")], "train")
        test_df = _make_df(spark, [], "test")

        cleaned, _ = ec.clean(train_df, test_df)
        row = cleaned.collect()[0]

        assert row["attack_category"] == "Other"

    def test_counts_and_removes_exact_duplicates(self, spark):
        dup_row = _default_row(attack_type="smurf")
        train_df = _make_df(spark, [dup_row, dup_row, dup_row], "train")
        test_df = _make_df(spark, [], "test")

        cleaned, duplicates_removed = ec.clean(train_df, test_df)

        assert duplicates_removed == 2
        assert cleaned.count() == 1

    def test_identical_rows_in_different_splits_are_not_deduplicated(self, spark):
        # same feature values appearing once in train and once in test is
        # expected (both files are independently sampled), not a duplicate
        row = _default_row(attack_type="normal")
        train_df = _make_df(spark, [row], "train")
        test_df = _make_df(spark, [row], "test")

        cleaned, duplicates_removed = ec.clean(train_df, test_df)

        assert duplicates_removed == 0
        assert cleaned.count() == 2
