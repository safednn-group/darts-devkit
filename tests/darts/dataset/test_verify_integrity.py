from types import SimpleNamespace

import pytest

from darts import DARTS


@pytest.fixture
def files_with_checksums():
    return [
        SimpleNamespace(filename="file1", checksum="abc"),
        SimpleNamespace(filename="file2", checksum="def"),
    ]


def test_verify_integrity_all_good(monkeypatch, files_with_checksums):
    d = object.__new__(DARTS)
    d._sample_data = SimpleNamespace(all=lambda: [files_with_checksums[0]])

    monkeypatch.setattr(d, "_progress", lambda iterable, *_: iterable)
    monkeypatch.setattr(d, "_get_checksum", lambda filename: "abc")
    d.verify_integrity()


def test_verify_integrity_bad_checksum(monkeypatch, files_with_checksums):
    d = object.__new__(DARTS)
    d._sample_data = SimpleNamespace(all=lambda: [files_with_checksums[0]])

    monkeypatch.setattr(d, "_progress", lambda iterable, *_: iterable)
    monkeypatch.setattr(d, "_get_checksum", lambda filename: "wrong")

    with pytest.raises(ValueError):
        d.verify_integrity()


def test_verify_integrity_multiple_bad(monkeypatch, files_with_checksums):
    d = object.__new__(DARTS)
    d._sample_data = SimpleNamespace(all=lambda: files_with_checksums)

    monkeypatch.setattr(d, "_progress", lambda iterable, *_: iterable)
    monkeypatch.setattr(d, "_get_checksum", lambda filename: "wrong")

    with pytest.raises(ValueError):
        d.verify_integrity()
