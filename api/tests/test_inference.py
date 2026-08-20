import pytest

import inference


class TestBuildFeaturesNslKdd:
    def test_unspecified_fields_fall_back_to_defaults(self):
        features, rule_flags = inference.build_features("nsl_kdd", {})

        assert features["protocol_type"] == "tcp"
        assert features["service"] == "http"
        assert features["flag"] == "SF"
        assert features["duration"] == 0.0
        assert rule_flags == []

    def test_provided_fields_override_defaults(self):
        features, _ = inference.build_features("nsl_kdd", {"protocol_type": "udp", "src_bytes": 500})

        assert features["protocol_type"] == "udp"
        assert features["src_bytes"] == 500.0

    def test_rejects_unknown_feature_key(self):
        with pytest.raises(ValueError, match="not_a_real_feature"):
            inference.build_features("nsl_kdd", {"not_a_real_feature": 1})

    def test_derives_failed_logins_rule_flag(self):
        features, rule_flags = inference.build_features("nsl_kdd", {"num_failed_logins": 5})
        assert "failed_logins" in rule_flags
        assert features["rule_flag_failed_logins"] == 1

    def test_derives_root_privilege_rule_flag_from_either_signal(self):
        _, flags_a = inference.build_features("nsl_kdd", {"root_shell": 1})
        _, flags_b = inference.build_features("nsl_kdd", {"num_root": 3})
        assert "root_privilege" in flags_a
        assert "root_privilege" in flags_b

    def test_derives_high_connection_rate_rule_flag(self):
        features, rule_flags = inference.build_features("nsl_kdd", {"count": 150})
        assert "high_connection_rate" in rule_flags
        assert features["rule_flag_high_connection_rate"] == 1

    def test_land_attack_and_error_rate_flags(self):
        _, flags_a = inference.build_features("nsl_kdd", {"land": 1})
        _, flags_b = inference.build_features("nsl_kdd", {"serror_rate": 0.9})
        assert "land_attack" in flags_a
        assert "high_error_rate" in flags_b

    def test_no_flags_triggered_on_defaults(self):
        _, rule_flags = inference.build_features("nsl_kdd", {})
        assert rule_flags == []


class TestBuildFeaturesCicids2017:
    def test_unspecified_fields_default_to_zero(self):
        features, rule_flags = inference.build_features("cicids2017", {})
        assert features["destination_port"] == 0.0
        # everything defaults to 0: both payload lengths (zero_payload) and
        # total_backward_packets (asymmetric_flow)
        assert set(rule_flags) == {"zero_payload", "asymmetric_flow"}

    def test_rejects_unknown_feature_key(self):
        with pytest.raises(ValueError, match="not_a_real_feature"):
            inference.build_features("cicids2017", {"not_a_real_feature": 1})

    def test_malicious_port_rule_flag(self):
        features, rule_flags = inference.build_features(
            "cicids2017", {"destination_port": 4444, "total_length_of_fwd_packets": 10, "total_length_of_bwd_packets": 10}
        )
        assert "malicious_port" in rule_flags
        assert "zero_payload" not in rule_flags

    def test_syn_flood_like_rule_flag(self):
        features, rule_flags = inference.build_features("cicids2017", {"syn_flag_count": 1, "ack_flag_count": 0})
        assert "syn_flood_like" in rule_flags

    def test_asymmetric_flow_rule_flag(self):
        features, rule_flags = inference.build_features("cicids2017", {"total_backward_packets": 0})
        assert "asymmetric_flow" in rule_flags


class TestBuildFeaturesInvalidDataset:
    def test_rejects_unknown_dataset(self):
        with pytest.raises(ValueError, match="dataset must be one of"):
            inference.build_features("not_a_real_dataset", {})


class TestScore:
    def test_formats_model_output_into_response_dict(self, monkeypatch):
        fake_prediction = {"probability": [0.13, 0.87], "prediction": 1}

        class FakeDF:
            def select(self, *a, **k):
                return self

            def first(self):
                return fake_prediction

        class FakeModel:
            def transform(self, df):
                return FakeDF()

        class FakeSpark:
            def createDataFrame(self, rows):
                return object()

        monkeypatch.setattr(inference, "_get_spark", lambda: FakeSpark())
        monkeypatch.setattr(inference, "_get_model", lambda dataset: FakeModel())

        result = inference.score("nsl_kdd", {})
        assert result["dataset"] == "nsl_kdd"
        assert result["attack_probability"] == 0.87
        assert result["predicted_label"] == 1
        assert "features_used" in result
