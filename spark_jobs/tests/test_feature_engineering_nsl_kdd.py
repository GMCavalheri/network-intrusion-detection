import feature_engineering_nsl_kdd as fe


def _row(**overrides):
    base = {
        "num_failed_logins": 0.0, "land": 0.0, "root_shell": 0.0, "num_root": 0.0,
        "count": 1.0, "serror_rate": 0.0, "rerror_rate": 0.0,
    }
    base.update(overrides)
    return base


class TestEngineerFeatures:
    def test_flags_repeated_failed_logins(self, spark):
        df = spark.createDataFrame([_row(num_failed_logins=5.0), _row(num_failed_logins=1.0)])
        out = fe.engineer_features(df).collect()

        flags = {r["num_failed_logins"]: r["rule_flag_failed_logins"] for r in out}
        assert flags[5.0] == 1
        assert flags[1.0] == 0

    def test_flags_land_attack(self, spark):
        df = spark.createDataFrame([_row(land=1.0), _row(land=0.0)])
        out = fe.engineer_features(df).collect()

        flags = {r["land"]: r["rule_flag_land_attack"] for r in out}
        assert flags[1.0] == 1
        assert flags[0.0] == 0

    def test_flags_root_privilege_from_either_signal(self, spark):
        df = spark.createDataFrame([
            _row(root_shell=1.0, num_root=0.0),
            _row(root_shell=0.0, num_root=3.0),
            _row(root_shell=0.0, num_root=0.0),
        ])
        out = fe.engineer_features(df).collect()

        assert out[0]["rule_flag_root_privilege"] == 1
        assert out[1]["rule_flag_root_privilege"] == 1
        assert out[2]["rule_flag_root_privilege"] == 0

    def test_flags_high_connection_rate(self, spark):
        df = spark.createDataFrame([_row(count=150.0), _row(count=10.0)])
        out = fe.engineer_features(df).collect()

        flags = {r["count"]: r["rule_flag_high_connection_rate"] for r in out}
        assert flags[150.0] == 1
        assert flags[10.0] == 0

    def test_flags_high_error_rate_from_either_serror_or_rerror(self, spark):
        df = spark.createDataFrame([
            _row(serror_rate=0.9, rerror_rate=0.0),
            _row(serror_rate=0.0, rerror_rate=0.9),
            _row(serror_rate=0.1, rerror_rate=0.1),
        ])
        out = fe.engineer_features(df).collect()

        assert out[0]["rule_flag_high_error_rate"] == 1
        assert out[1]["rule_flag_high_error_rate"] == 1
        assert out[2]["rule_flag_high_error_rate"] == 0

    def test_rule_flags_joins_only_triggered_flags(self, spark):
        df = spark.createDataFrame([_row(num_failed_logins=5.0, land=1.0)])
        row = fe.engineer_features(df).collect()[0]

        flags = set(row["rule_flags"].split(","))
        assert flags == {"failed_logins", "land_attack"}

    def test_rule_flags_empty_string_when_nothing_triggered(self, spark):
        df = spark.createDataFrame([_row()])
        row = fe.engineer_features(df).collect()[0]

        assert row["rule_flags"] == ""
