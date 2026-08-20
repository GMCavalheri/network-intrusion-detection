import json
import random

import train_model as tm


def _synthetic_df(spark, n_per_class=25, category_col=None, seed=0):
    """A small, roughly-linearly-separable binary classification frame:
    class 1 rows have systematically higher feature_a/feature_b than class 0,
    so GBTClassifier has something real to learn within a tiny test fixture."""
    rng = random.Random(seed)
    rows = []
    for label in (0, 1):
        base = 10.0 if label == 1 else 0.0
        for i in range(n_per_class):
            row = {
                "feature_a": base + rng.uniform(0, 3),
                "feature_b": base + rng.uniform(0, 3),
                "rule_flag_something": 1 if label == 1 and i % 2 == 0 else 0,
                "is_attack": label,
                "attack_category": "DoS" if label == 1 else "Benign",
            }
            if category_col:
                row[category_col] = "tcp" if i % 2 == 0 else "udp"
            rows.append(row)
    return spark.createDataFrame(rows)


class TestRuleFlagColumns:
    def test_finds_only_rule_flag_prefixed_columns(self):
        class _FakeDf:
            columns = ["amount", "rule_flag_x", "rule_flag_y", "category"]

        assert tm._rule_flag_columns(_FakeDf()) == ["rule_flag_x", "rule_flag_y"]

    def test_empty_when_none_present(self):
        class _FakeDf:
            columns = ["amount", "category"]

        assert tm._rule_flag_columns(_FakeDf()) == []


class TestThresholdCurve:
    def test_recall_is_monotonically_non_increasing_as_threshold_rises(self, spark):
        predictions = spark.createDataFrame([
            {"is_attack": 1, "attack_probability": 0.9},
            {"is_attack": 1, "attack_probability": 0.6},
            {"is_attack": 0, "attack_probability": 0.4},
            {"is_attack": 0, "attack_probability": 0.1},
        ])

        curve = tm._threshold_curve(predictions, resolution=0.1)
        by_threshold = sorted(curve, key=lambda p: p["threshold"])  # ascending threshold

        recalls = [p["recall"] for p in by_threshold]
        assert recalls == sorted(recalls, reverse=True)  # recall falls (or holds) as the threshold rises

    def test_lowest_threshold_catches_every_positive(self, spark):
        predictions = spark.createDataFrame([
            {"is_attack": 1, "attack_probability": 0.9},
            {"is_attack": 1, "attack_probability": 0.05},
            {"is_attack": 0, "attack_probability": 0.5},
        ])

        curve = tm._threshold_curve(predictions, resolution=0.1)
        lowest = min(curve, key=lambda p: p["threshold"])

        assert lowest["recall"] == 1.0

    def test_matches_default_threshold_confusion_matrix_style_precision(self, spark):
        # a hand-checkable case: at threshold exactly 0.5, 1 true positive,
        # 1 false positive, precision must be 0.5
        predictions = spark.createDataFrame([
            {"is_attack": 1, "attack_probability": 0.5},
            {"is_attack": 0, "attack_probability": 0.5},
            {"is_attack": 0, "attack_probability": 0.1},
        ])

        curve = tm._threshold_curve(predictions, resolution=0.1)
        at_half = next(p for p in curve if abs(p["threshold"] - 0.5) < 1e-9)

        assert at_half["precision"] == 0.5
        assert at_half["recall"] == 1.0


class TestTrainAndEvaluate:
    def test_returns_model_and_expected_metric_keys(self, spark):
        train_df = _synthetic_df(spark, seed=1)
        test_df = _synthetic_df(spark, seed=2)

        model, metrics = tm.train_and_evaluate(train_df, test_df, ["feature_a", "feature_b"], [])

        assert hasattr(model, "transform")  # a fitted PipelineModel
        for key in (
            "auc_roc", "auc_pr", "precision", "recall", "f1", "confusion_matrix",
            "feature_importances", "feature_columns", "category_breakdown",
            "model_type", "train_rows", "test_rows", "attack_rate_train", "attack_rate_test",
        ):
            assert key in metrics, f"missing metric key: {key}"
        assert metrics["model_type"] == "GBTClassifier"
        assert metrics["test_rows"] == 50
        # rule_flag_* columns get folded into the feature set automatically
        assert "rule_flag_something" in metrics["feature_columns"]

    def test_metrics_are_json_serializable(self, spark):
        train_df = _synthetic_df(spark, seed=1)
        test_df = _synthetic_df(spark, seed=2)

        _, metrics = tm.train_and_evaluate(train_df, test_df, ["feature_a", "feature_b"], [])

        json.dumps(metrics)  # must not raise

    def test_learns_the_separable_signal_reasonably_well(self, spark):
        # not a claim about GBT hyperparameters - just a sanity check that
        # the pipeline (indexer/assembler/classifier wiring) actually works
        # end-to-end on data with a real signal, not just that it doesn't crash
        train_df = _synthetic_df(spark, n_per_class=40, seed=1)
        test_df = _synthetic_df(spark, n_per_class=40, seed=2)

        _, metrics = tm.train_and_evaluate(train_df, test_df, ["feature_a", "feature_b"], [])

        assert metrics["auc_roc"] > 0.8

    def test_handles_categorical_features_via_string_indexer(self, spark):
        train_df = _synthetic_df(spark, category_col="protocol_type", seed=1)
        test_df = _synthetic_df(spark, category_col="protocol_type", seed=2)

        model, metrics = tm.train_and_evaluate(
            train_df, test_df, ["feature_a", "feature_b"], ["protocol_type"]
        )

        assert "protocol_type_idx" in metrics["feature_columns"]

    def test_category_breakdown_counts_match_input(self, spark):
        train_df = _synthetic_df(spark, seed=1)
        test_df = _synthetic_df(spark, n_per_class=10, seed=2)

        _, metrics = tm.train_and_evaluate(train_df, test_df, ["feature_a", "feature_b"], [])

        counts = {row["attack_category"]: row["count"] for row in metrics["category_breakdown"]}
        assert counts == {"Benign": 10, "DoS": 10}
