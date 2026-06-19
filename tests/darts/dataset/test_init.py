import sys
from pathlib import Path

import darts_devkit
from darts_devkit import DARTS


def test_init_sets_root_and_version(monkeypatch):
    monkeypatch.setattr(DARTS, "_load_table", lambda self, name, cls: None)
    monkeypatch.setattr(DARTS, "_create_relationships", lambda self: None)

    d = DARTS("data", "v1")
    assert d._root == Path("data")
    assert d._version == "v1"


def test_init_show_progress(monkeypatch):
    monkeypatch.setattr(DARTS, "_load_table", lambda self, name, cls: None)
    monkeypatch.setattr(DARTS, "_create_relationships", lambda self: None)

    def make_instance(isatty_return, logger_return):
        monkeypatch.setattr(sys.stderr, "isatty", lambda: isatty_return)
        monkeypatch.setattr(darts_devkit.dataset.darts.logger, "isEnabledFor", lambda level: logger_return)
        return DARTS("data", "v1")

    # Test all combinations
    d = make_instance(True, True)
    assert d._show_progress is True

    d = make_instance(True, False)
    assert d._show_progress is False

    d = make_instance(False, True)
    assert d._show_progress is False

    d = make_instance(False, False)
    assert d._show_progress is False


def test_repr(monkeypatch):
    monkeypatch.setattr(DARTS, "_load_table", lambda self, name, cls: None)
    monkeypatch.setattr(DARTS, "_create_relationships", lambda self: None)

    d = DARTS("data", "v1")
    r = repr(d)
    assert "DARTS(root=data" in r and "version=v1" in r


def test_load_table_calls(monkeypatch):
    called_tables = []

    def fake_load_table(self, name, cls):
        called_tables.append(name)

    monkeypatch.setattr(DARTS, "_load_table", fake_load_table)
    monkeypatch.setattr(DARTS, "_create_relationships", lambda self: None)

    DARTS("data", "v1")

    expected_tables = [
        "calibrated_sensor",
        "category",
        "ego_pose",
        "ins",
        "instance",
        "instance_2d",
        "metadata",
        "sample",
        "sample_annotation",
        "sample_annotation_2d",
        "sample_data",
        "scene",
        "sensor",
        "attribute",
    ]
    assert called_tables == expected_tables
