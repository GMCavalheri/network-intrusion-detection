import json

import db
import inference
import main

MINIMAL_FLOW_ROW = {
    "flow_id": "nsl_kdd-1", "dataset_source": "nsl_kdd", "split": "test",
    "source_day": None, "protocol": "tcp", "duration": 0.0, "bytes_sent": 491.0,
    "bytes_received": 0.0, "is_attack_actual": 0, "attack_category_actual": "Benign",
    "predicted_label": 0, "attack_probability": 0.02, "rule_flags": "",
    "raw_features": {"protocol_type": "tcp"}, "scored_at": "2026-01-01T10:00:00",
}


class TestHealthAndMeta:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_meta_returns_both_datasets_reference_data(self, client):
        resp = client.get("/meta")
        body = resp.json()

        assert body["datasets"] == ["nsl_kdd", "cicids2017"]
        assert "Benign" in body["attack_categories"]
        assert "tcp" in body["nsl_kdd"]["protocol_types"]
        assert "http" in body["nsl_kdd"]["services"]
        assert "destination_port" in body["cicids2017"]["form_fields"]


class TestFlows:
    def test_list_flows_paginates_and_passes_filters(self, client, monkeypatch):
        captured = {}

        def fake_fetch(**kwargs):
            captured.update(kwargs)
            return [dict(MINIMAL_FLOW_ROW)], 1

        monkeypatch.setattr(db, "fetch_flows", fake_fetch)
        resp = client.get("/flows", params={"limit": 10, "dataset": "nsl_kdd", "predicted_label": 0})

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["flow_id"] == "nsl_kdd-1"
        assert captured["dataset"] == "nsl_kdd"
        assert captured["predicted_label"] == 0
        assert captured["limit"] == 10

    def test_list_flows_rejects_unknown_dataset(self, client):
        resp = client.get("/flows", params={"dataset": "not_a_real_dataset"})
        assert resp.status_code == 422

    def test_get_flow_404_when_missing(self, client, monkeypatch):
        monkeypatch.setattr(db, "fetch_flow_by_id", lambda flow_id: None)
        resp = client.get("/flows/DOES-NOT-EXIST")
        assert resp.status_code == 404

    def test_get_flow_found(self, client, monkeypatch):
        monkeypatch.setattr(db, "fetch_flow_by_id", lambda flow_id: dict(MINIMAL_FLOW_ROW))
        resp = client.get("/flows/nsl_kdd-1")
        assert resp.status_code == 200
        assert resp.json()["flow_id"] == "nsl_kdd-1"


class TestStats:
    def test_summary_requires_dataset_param(self, client):
        resp = client.get("/stats/summary")
        assert resp.status_code == 422

    def test_summary_rejects_unknown_dataset(self, client):
        resp = client.get("/stats/summary", params={"dataset": "bogus"})
        assert resp.status_code == 422

    def test_summary_404_when_no_data_yet(self, client, monkeypatch):
        monkeypatch.setattr(db, "fetch_summary_stats", lambda dataset: {"total_flows": 0})
        resp = client.get("/stats/summary", params={"dataset": "nsl_kdd"})
        assert resp.status_code == 404

    def test_summary_includes_model_metrics_when_present(self, client, monkeypatch, tmp_path):
        stats = {
            "total_flows": 100, "actual_attack_count": 40, "predicted_attack_count": 38,
            "avg_attack_probability": 0.3,
        }
        monkeypatch.setattr(db, "fetch_summary_stats", lambda dataset: dict(stats))

        metrics_file = tmp_path / "nsl_kdd_metrics.json"
        metrics_file.write_text(json.dumps({"auc_roc": 0.93}))
        monkeypatch.setitem(main.METRICS_PATHS, "nsl_kdd", str(metrics_file))

        resp = client.get("/stats/summary", params={"dataset": "nsl_kdd"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_metrics"] == {"auc_roc": 0.93}
        assert body["dataset_source"] == "nsl_kdd"

    def test_datasets_and_categories_pass_through(self, client, monkeypatch):
        monkeypatch.setattr(db, "fetch_dataset_stats", lambda: [])
        monkeypatch.setattr(db, "fetch_category_breakdown", lambda dataset=None: [])
        assert client.get("/stats/datasets").json() == []
        assert client.get("/stats/categories").json() == []

    def test_categories_rejects_unknown_dataset(self, client):
        resp = client.get("/stats/categories", params={"dataset": "bogus"})
        assert resp.status_code == 422


class TestDataQualityReport:
    def test_requires_dataset_param(self, client):
        resp = client.get("/reports/data-quality")
        assert resp.status_code == 422

    def test_404_when_report_missing(self, client, monkeypatch, tmp_path):
        monkeypatch.setitem(main.DQ_REPORT_PATHS, "nsl_kdd", str(tmp_path / "nope.json"))
        resp = client.get("/reports/data-quality", params={"dataset": "nsl_kdd"})
        assert resp.status_code == 404

    def test_returns_report_contents_when_present(self, client, monkeypatch, tmp_path):
        report_file = tmp_path / "dq.json"
        report_file.write_text(json.dumps({"duplicates_removed": 42}))
        monkeypatch.setitem(main.DQ_REPORT_PATHS, "cicids2017", str(report_file))
        resp = client.get("/reports/data-quality", params={"dataset": "cicids2017"})
        assert resp.status_code == 200
        assert resp.json() == {"duplicates_removed": 42}


class TestScore:
    def test_successful_score_delegates_to_inference(self, client, monkeypatch):
        canned = {
            "dataset": "nsl_kdd", "attack_probability": 0.42, "predicted_label": 0,
            "rule_flags": [], "features_used": {},
        }
        monkeypatch.setattr(inference, "score", lambda dataset, features: canned)
        resp = client.post("/score", json={"dataset": "nsl_kdd", "features": {"duration": 5}})
        assert resp.status_code == 200
        assert resp.json() == canned

    def test_unknown_feature_key_returns_422_not_a_crash(self, client, monkeypatch):
        def boom(dataset, features):
            raise ValueError("Unknown NSL-KDD feature(s): ['not_a_real_feature']")

        monkeypatch.setattr(inference, "score", boom)
        resp = client.post("/score", json={"dataset": "nsl_kdd", "features": {"not_a_real_feature": 1}})
        assert resp.status_code == 422

    def test_scoring_failure_returns_503_not_a_crash(self, client, monkeypatch):
        def boom(dataset, features):
            raise RuntimeError("model not loaded")

        monkeypatch.setattr(inference, "score", boom)
        resp = client.post("/score", json={"dataset": "nsl_kdd", "features": {}})
        assert resp.status_code == 503
