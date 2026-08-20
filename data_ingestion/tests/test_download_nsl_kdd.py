import io
import logging
import os

import download_nsl_kdd as dnk


def _fake_download_file_factory(row_counts):
    """Returns a stand-in for common.download_file that writes `row_counts[filename]`
    newline-terminated rows to dest_path instead of hitting the network."""

    def _fake(url, dest_path, logger, session=None):
        filename = os.path.basename(dest_path)
        n_rows = row_counts[filename]
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "w") as f:
            for i in range(n_rows):
                f.write(f"row-{i}\n")
        return 0

    return _fake


class TestDownloadNslKdd:
    def test_builds_url_with_url_encoded_plus(self, tmp_path, monkeypatch):
        requested_urls = []

        def fake_download_file(url, dest_path, logger, session=None):
            requested_urls.append(url)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "w") as f:
                f.write("row\n" * dnk.EXPECTED_ROW_COUNTS[os.path.basename(dest_path)])

        monkeypatch.setattr(dnk, "download_file", fake_download_file)

        dnk.download_nsl_kdd(str(tmp_path), base_url="https://example.com/mirror")

        assert "https://example.com/mirror/KDDTrain%2B.txt" in requested_urls
        assert "https://example.com/mirror/KDDTest%2B.txt" in requested_urls

    def test_returns_dest_paths_for_both_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            dnk, "download_file", _fake_download_file_factory(dnk.EXPECTED_ROW_COUNTS)
        )

        results = dnk.download_nsl_kdd(str(tmp_path))

        assert set(results.keys()) == {"KDDTrain+.txt", "KDDTest+.txt"}
        assert results["KDDTrain+.txt"] == os.path.join(str(tmp_path), "KDDTrain+.txt")
        assert os.path.exists(results["KDDTrain+.txt"])

    def test_warns_but_does_not_raise_on_row_count_mismatch(self, tmp_path, monkeypatch):
        # dnk.logger has propagate=False (console + rotating-file handler
        # pattern shared across the project, see common.get_logger), so
        # neither caplog nor capsys/capfd reliably observes it - attach a
        # throwaway handler directly instead.
        wrong_counts = {"KDDTrain+.txt": 10, "KDDTest+.txt": 5}
        monkeypatch.setattr(dnk, "download_file", _fake_download_file_factory(wrong_counts))

        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        dnk.logger.addHandler(handler)
        try:
            results = dnk.download_nsl_kdd(str(tmp_path))  # must not raise
        finally:
            dnk.logger.removeHandler(handler)

        assert "expected" in buf.getvalue()
        assert len(results) == 2
