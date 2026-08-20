import os

import etl_clean_cicids2017 as ec
from common import CICIDS2017_RAW_COLUMNS
from pyspark.sql import functions as F


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _default_row(**overrides):
    """A full 79-column CICIDS2017 row (78 raw features + label), defaulted
    to 0.0/"BENIGN" so tests only need to specify the columns they care
    about. Mirrors _default_row in test_etl_clean_nsl_kdd.py."""
    row = {c: 0.0 for c in CICIDS2017_RAW_COLUMNS}
    row["destination_port"] = 80
    row["flow_duration"] = 1000
    row["label"] = "BENIGN"
    row.update(overrides)
    return tuple(row[c] for c in CICIDS2017_RAW_COLUMNS)


def _make_df(spark, rows, day="monday"):
    df = spark.createDataFrame(list(rows), schema=ec._build_schema())
    split = "train" if day in ("monday", "wednesday") else "test"
    return df.drop("fwd_header_length_dup").withColumn("source_day", F.lit(day)).withColumn("split", F.lit(split))


class TestReadDay:
    def test_reads_real_shaped_header_positionally(self, spark, tmp_path):
        # Real CICIDS2017 headers have leading spaces and a duplicate
        # "Fwd Header Length" column (verified against a downloaded file) -
        # this fixture reproduces that exact shape.
        path = os.path.join(str(tmp_path), "Monday-WorkingHours.pcap_ISCX.csv")
        _write(
            path,
            " Destination Port, Flow Duration," + ",".join(["0"] * 76) + ", Label\n"
            "443,5000," + ",".join(["1"] * 76) + ",BENIGN\n",
        )

        df = ec.read_day(spark, path, "monday")
        row = df.collect()[0]

        assert row["destination_port"] == 443
        assert row["flow_duration"] == 5000
        assert row["label"] == "BENIGN"
        assert row["source_day"] == "monday"
        assert row["split"] == "train"
        assert "fwd_header_length_dup" not in df.columns

    def test_friday_is_tagged_as_test_split(self, spark, tmp_path):
        path = os.path.join(str(tmp_path), "Friday.csv")
        # header line content is irrelevant (read positionally) - just needs a first line to skip
        _write(path, ("h," * 78) + "h\n" + ("0," * 78) + "BENIGN\n")

        df = ec.read_day(spark, path, "friday_ddos")
        assert df.collect()[0]["split"] == "test"


class TestClean:
    def test_drops_negative_duration_rows(self, spark):
        df = _make_df(spark, [_default_row(flow_duration=-1), _default_row(flow_duration=100)])

        cleaned, stats = ec.clean(df)

        assert stats["negative_duration_rows_dropped"] == 1
        assert cleaned.count() == 1
        assert cleaned.collect()[0]["flow_duration"] == 100

    def test_nulls_out_infinity_and_nan_rate_values(self, spark):
        df = _make_df(spark, [
            _default_row(flow_bytes_per_s=float("inf")),
            _default_row(flow_bytes_per_s=float("nan"), destination_port=81),
            _default_row(flow_bytes_per_s=500.0, destination_port=82),
        ])

        cleaned, stats = ec.clean(df)
        rows = {r["destination_port"]: r["flow_bytes_per_s"] for r in cleaned.collect()}

        assert stats["rate_column_infinity_or_nan_nulled"]["flow_bytes_per_s"] == 2
        assert rows[80] is None
        assert rows[81] is None
        assert rows[82] == 500.0

    def test_removes_exact_duplicate_flows(self, spark):
        dup = _default_row(destination_port=443)
        df = _make_df(spark, [dup, dup, dup, _default_row(destination_port=8080)])

        cleaned, stats = ec.clean(df)

        assert stats["duplicates_removed"] == 2
        assert cleaned.count() == 2

    def test_maps_label_to_unified_category(self, spark):
        df = _make_df(spark, [
            _default_row(label="BENIGN", destination_port=1),
            _default_row(label="DDoS", destination_port=2),
            _default_row(label="PortScan", destination_port=3),
        ])

        cleaned, _ = ec.clean(df)
        rows = {r["destination_port"]: r for r in cleaned.collect()}

        assert rows[1]["is_attack"] == 0
        assert rows[1]["attack_category"] == "Benign"
        assert rows[2]["is_attack"] == 1
        assert rows[2]["attack_category"] == "DoS"
        assert rows[3]["attack_category"] == "Probe/Scan"

    def test_unmapped_label_falls_back_to_other(self, spark):
        df = _make_df(spark, [_default_row(label="SomeFutureAttackType")])

        cleaned, _ = ec.clean(df)

        assert cleaned.collect()[0]["attack_category"] == "Other"
