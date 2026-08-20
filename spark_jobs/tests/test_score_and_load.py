import json

import score_and_load as sal
from pyspark.sql import functions as F


def _nsl_kdd_scored_df(spark):
    return spark.createDataFrame([
        {
            "protocol_type": "tcp", "service": "http", "duration": 5.0,
            "src_bytes": 100.0, "dst_bytes": 200.0, "is_attack": 0,
            "attack_category": "Benign", "prediction": 0.0, "attack_probability": 0.12,
            "rule_flags": "", "split": "test",
        },
        {
            "protocol_type": "tcp", "service": "private", "duration": 0.0,
            "src_bytes": 0.0, "dst_bytes": 0.0, "is_attack": 1,
            "attack_category": "DoS", "prediction": 1.0, "attack_probability": 0.91,
            "rule_flags": "high_connection_rate", "split": "test",
        },
    ])


def _cicids2017_scored_df(spark):
    return spark.createDataFrame([
        {
            "destination_port": 80, "flow_duration": 5000.0,
            "total_length_of_fwd_packets": 300.0, "total_length_of_bwd_packets": 150.0,
            "is_attack": 0, "attack_category": "Benign", "prediction": 0.0,
            "attack_probability": 0.05, "rule_flags": "", "split": "train", "source_day": "monday",
        },
        {
            "destination_port": 443, "flow_duration": 12.0,
            "total_length_of_fwd_packets": 0.0, "total_length_of_bwd_packets": 0.0,
            "is_attack": 1, "attack_category": "DoS", "prediction": 1.0,
            "attack_probability": 0.97, "rule_flags": "zero_payload", "split": "test", "source_day": "friday_ddos",
        },
    ])


class TestProjectToCommonSchema:
    def test_maps_nsl_kdd_dataset_specific_columns_onto_common_schema(self, spark):
        df = _nsl_kdd_scored_df(spark)

        out = sal.project_to_common_schema(df, "nsl_kdd", ["protocol_type", "service", "duration"])
        rows = {r["attack_category_actual"]: r for r in out.collect()}

        assert set(out.columns) == set(sal.COMMON_COLUMNS)
        assert rows["Benign"]["protocol"] == "tcp"
        assert rows["Benign"]["bytes_sent"] == 100.0
        assert rows["Benign"]["bytes_received"] == 200.0
        assert rows["Benign"]["duration"] == 5.0
        assert rows["Benign"]["source_day"] is None  # NSL-KDD has no capture-day concept
        assert rows["Benign"]["dataset_source"] == "nsl_kdd"
        assert rows["Benign"]["predicted_label"] == 0
        assert rows["DoS"]["predicted_label"] == 1

    def test_maps_cicids2017_dataset_specific_columns_onto_common_schema(self, spark):
        df = _cicids2017_scored_df(spark)

        out = sal.project_to_common_schema(df, "cicids2017", ["destination_port", "flow_duration"])
        rows = {r["attack_category_actual"]: r for r in out.collect()}

        assert set(out.columns) == set(sal.COMMON_COLUMNS)
        assert rows["Benign"]["protocol"] is None  # this CSV release has no protocol column
        assert rows["Benign"]["bytes_sent"] == 300.0
        assert rows["Benign"]["bytes_received"] == 150.0
        assert rows["Benign"]["duration"] == 5000.0
        assert rows["Benign"]["source_day"] == "monday"
        assert rows["DoS"]["source_day"] == "friday_ddos"
        assert rows["Benign"]["dataset_source"] == "cicids2017"

    def test_flow_id_is_prefixed_with_dataset_source_and_unique(self, spark):
        df = _nsl_kdd_scored_df(spark)

        out = sal.project_to_common_schema(df, "nsl_kdd", ["duration"])
        flow_ids = [r["flow_id"] for r in out.collect()]

        assert all(fid.startswith("nsl_kdd-") for fid in flow_ids)
        assert len(set(flow_ids)) == len(flow_ids)

    def test_raw_features_is_valid_json_containing_the_requested_columns(self, spark):
        df = _nsl_kdd_scored_df(spark)

        out = sal.project_to_common_schema(df, "nsl_kdd", ["protocol_type", "service"])
        row = out.filter(F.col("attack_category_actual") == "DoS").collect()[0]
        parsed = json.loads(row["raw_features"])

        assert parsed == {"protocol_type": "tcp", "service": "private"}

    def test_two_datasets_can_be_unioned_after_projection(self, spark):
        nsl = sal.project_to_common_schema(_nsl_kdd_scored_df(spark), "nsl_kdd", ["duration"])
        cic = sal.project_to_common_schema(_cicids2017_scored_df(spark), "cicids2017", ["destination_port"])

        combined = nsl.unionByName(cic)  # must not raise (schemas must line up)

        assert combined.count() == 4
        assert set(r["dataset_source"] for r in combined.collect()) == {"nsl_kdd", "cicids2017"}
