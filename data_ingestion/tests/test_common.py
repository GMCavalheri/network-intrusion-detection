import logging
import os

import pytest

from common import download_file

logger = logging.getLogger("test")


class FakeResponse:
    def __init__(self, chunks, status=200):
        self._chunks = chunks
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"status {self.status_code}")

    def iter_content(self, chunk_size=None):
        yield from self._chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, chunks=(b"hello ", b"world"), status=200):
        self._chunks = chunks
        self._status = status
        self.requested_urls = []

    def get(self, url, stream=True, timeout=60):
        self.requested_urls.append(url)
        return FakeResponse(self._chunks, self._status)


class TestDownloadFile:
    def test_writes_full_content(self, tmp_path):
        dest = tmp_path / "out.txt"
        session = FakeSession(chunks=(b"hello ", b"world"))

        written = download_file("https://example.com/f.txt", str(dest), logger, session=session)

        assert dest.read_bytes() == b"hello world"
        assert written == len(b"hello world")
        assert session.requested_urls == ["https://example.com/f.txt"]

    def test_skips_when_file_already_exists_and_nonempty(self, tmp_path):
        dest = tmp_path / "out.txt"
        dest.write_bytes(b"already here")
        session = FakeSession()

        written = download_file("https://example.com/f.txt", str(dest), logger, session=session)

        assert written == 0
        assert dest.read_bytes() == b"already here"  # untouched
        assert session.requested_urls == []  # never made the request

    def test_redownloads_when_existing_file_is_empty(self, tmp_path):
        dest = tmp_path / "out.txt"
        dest.write_bytes(b"")
        session = FakeSession(chunks=(b"real content",))

        written = download_file("https://example.com/f.txt", str(dest), logger, session=session)

        assert written == len(b"real content")
        assert dest.read_bytes() == b"real content"

    def test_creates_parent_directories(self, tmp_path):
        dest = tmp_path / "nested" / "dir" / "out.txt"
        session = FakeSession(chunks=(b"data",))

        download_file("https://example.com/f.txt", str(dest), logger, session=session)

        assert dest.read_bytes() == b"data"

    def test_raises_on_http_error(self, tmp_path):
        import requests

        dest = tmp_path / "out.txt"
        session = FakeSession(status=404)

        with pytest.raises(requests.HTTPError):
            download_file("https://example.com/missing.txt", str(dest), logger, session=session)
        assert not dest.exists()

    def test_does_not_leave_partial_file_visible_under_final_name(self, tmp_path):
        # writes to a .part file and os.replace()s it, so a reader can never
        # observe a truncated file at the destination path mid-download
        dest = tmp_path / "out.txt"
        session = FakeSession(chunks=(b"x" * 10,))

        download_file("https://example.com/f.txt", str(dest), logger, session=session)

        assert not os.path.exists(str(dest) + ".part")
