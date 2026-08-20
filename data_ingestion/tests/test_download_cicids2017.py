import os

import pytest

import download_cicids2017 as dc


class TestResolveFiles:
    def test_maps_known_day_keys_to_upstream_filenames(self):
        files = dc.resolve_files(["monday", "friday_ddos"])

        assert files == {
            "monday": "Monday-WorkingHours.pcap_ISCX.csv",
            "friday_ddos": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        }

    def test_all_eight_days_are_mapped_and_unique(self):
        assert len(dc.DAY_FILES) == 8
        assert len(set(dc.DAY_FILES.values())) == 8  # no accidental duplicate filenames

    def test_unknown_day_key_raises(self):
        with pytest.raises(ValueError, match="nonexistent_day"):
            dc.resolve_files(["monday", "nonexistent_day"])


class TestDownloadCicids2017:
    def test_default_days_is_a_three_day_subset(self):
        assert dc.DEFAULT_DAYS == ["monday", "wednesday", "friday_ddos"]
        assert all(d in dc.DAY_FILES for d in dc.DEFAULT_DAYS)

    def test_downloads_each_requested_day_to_its_own_file(self, tmp_path, monkeypatch):
        requested = []

        def fake_download_file(url, dest_path, logger, session=None):
            requested.append((url, dest_path))

        monkeypatch.setattr(dc, "download_file", fake_download_file)

        results = dc.download_cicids2017(str(tmp_path), ["monday", "friday_ddos"], base_url="https://mirror.example.com")

        assert results["monday"] == os.path.join(str(tmp_path), "Monday-WorkingHours.pcap_ISCX.csv")
        assert results["friday_ddos"] == os.path.join(
            str(tmp_path), "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        )
        urls = [u for u, _ in requested]
        assert "https://mirror.example.com/Monday-WorkingHours.pcap_ISCX.csv" in urls
        assert "https://mirror.example.com/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" in urls

    def test_propagates_unknown_day_before_downloading_anything(self, tmp_path, monkeypatch):
        requested = []
        monkeypatch.setattr(dc, "download_file", lambda *a, **k: requested.append(a))

        with pytest.raises(ValueError):
            dc.download_cicids2017(str(tmp_path), ["monday", "bogus"])

        assert requested == []  # fails fast, no partial downloads
