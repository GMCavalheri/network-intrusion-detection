import feature_engineering_cicids2017 as fe


def _row(**overrides):
    base = {
        "total_length_of_fwd_packets": 100.0, "total_length_of_bwd_packets": 100.0,
        "syn_flag_count": 0.0, "ack_flag_count": 1.0,
        "destination_port": 80, "total_backward_packets": 2.0,
    }
    base.update(overrides)
    return base


class TestEngineerFeatures:
    def test_flags_zero_payload_flows(self, spark):
        df = spark.createDataFrame([
            _row(total_length_of_fwd_packets=0.0, total_length_of_bwd_packets=0.0, destination_port=1),
            _row(total_length_of_fwd_packets=50.0, total_length_of_bwd_packets=0.0, destination_port=2),
        ])
        out = {r["destination_port"]: r["rule_flag_zero_payload"] for r in fe.engineer_features(df).collect()}

        assert out[1] == 1
        assert out[2] == 0  # only one side is zero, not both

    def test_flags_syn_flood_like_flows(self, spark):
        df = spark.createDataFrame([
            _row(syn_flag_count=1.0, ack_flag_count=0.0, destination_port=1),
            _row(syn_flag_count=1.0, ack_flag_count=1.0, destination_port=2),
        ])
        out = {r["destination_port"]: r["rule_flag_syn_flood_like"] for r in fe.engineer_features(df).collect()}

        assert out[1] == 1  # SYN with no ACK - handshake never completed
        assert out[2] == 0  # normal completed handshake

    def test_flags_known_malicious_ports(self, spark):
        df = spark.createDataFrame([
            _row(destination_port=4444),
            _row(destination_port=443),
        ])
        out = {r["destination_port"]: r["rule_flag_malicious_port"] for r in fe.engineer_features(df).collect()}

        assert out[4444] == 1
        assert out[443] == 0

    def test_flags_asymmetric_flows(self, spark):
        df = spark.createDataFrame([
            _row(total_backward_packets=0.0, destination_port=1),
            _row(total_backward_packets=3.0, destination_port=2),
        ])
        out = {r["destination_port"]: r["rule_flag_asymmetric_flow"] for r in fe.engineer_features(df).collect()}

        assert out[1] == 1
        assert out[2] == 0

    def test_rule_flags_joins_only_triggered_flags(self, spark):
        df = spark.createDataFrame([_row(
            total_length_of_fwd_packets=0.0, total_length_of_bwd_packets=0.0, destination_port=4444,
        )])
        row = fe.engineer_features(df).collect()[0]

        flags = set(row["rule_flags"].split(","))
        assert flags == {"zero_payload", "malicious_port"}

    def test_rule_flags_empty_string_when_nothing_triggered(self, spark):
        df = spark.createDataFrame([_row()])
        row = fe.engineer_features(df).collect()[0]

        assert row["rule_flags"] == ""
