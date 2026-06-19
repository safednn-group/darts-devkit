from types import SimpleNamespace

from darts_devkit.dataset.darts import DARTS


def test_add_channel_to_sample_data(monkeypatch, sample_data_record, calibrated_sensor_record, sensor_record):
    d = object.__new__(DARTS)
    sd = sample_data_record()
    cs = calibrated_sensor_record()
    sensor = sensor_record()

    d._sample_data = SimpleNamespace(all=lambda: [sd])
    d._calibrated_sensor = SimpleNamespace(get=lambda token: cs)
    d._sensor = SimpleNamespace(get=lambda token: sensor)
    monkeypatch.setattr(d, "_progress", lambda iterable, *_: iterable)

    DARTS._add_channel_and_modality_to_sample_data(d)
    assert sd.channel == sensor.channel
    assert sd.modality == sensor.modality


def test_add_ins_to_sample(monkeypatch, sample_record, ins_record):
    d = object.__new__(DARTS)
    s = sample_record()
    i = ins_record()

    d._sample = SimpleNamespace(all=lambda: [s], get=lambda token: s)
    d._ins = SimpleNamespace(all=lambda: [i])
    monkeypatch.setattr(d, "_progress", lambda iterable, *_: iterable)

    DARTS._add_ins_to_sample(d)
    assert s.ins_token == i.token


def test_add_scene_metadata_to_scene(monkeypatch, scene_metadata_record, scene_record):
    d = object.__new__(DARTS)
    meta = scene_metadata_record()
    scene = scene_record()

    d._scene_metadata = SimpleNamespace(all=lambda: [meta])
    d._scene = SimpleNamespace(get=lambda token: scene)
    monkeypatch.setattr(d, "_progress", lambda iterable, *_: iterable)

    DARTS._add_scene_metadata_to_scene(d)
    assert scene.scene_metadata_token == meta.token


def test_add_data_to_sample(monkeypatch, sample_data_record, sample_record):
    d = object.__new__(DARTS)
    sd = sample_data_record(channel="CAM_FRONT")
    s = sample_record()

    d._sample_data = SimpleNamespace(all=lambda: [sd])
    d._sample = SimpleNamespace(all=lambda: [s])
    monkeypatch.setattr(d, "_progress", lambda iterable, *_: iterable)

    DARTS._add_data_to_sample(d)
    assert s.data == {sd.channel: sd.token}


def test_add_annotations_to_sample(monkeypatch, sample_annotation_record, sample_record):
    d = object.__new__(DARTS)
    ann = sample_annotation_record()
    s = sample_record()

    d._sample_annotation = SimpleNamespace(all=lambda: [ann])
    d._sample = SimpleNamespace(all=lambda: [s])
    monkeypatch.setattr(d, "_progress", lambda iterable, *_: iterable)

    DARTS._add_annotations_to_sample(d)
    assert s.anns == (ann.token,)


def test_add_annotations_to_sample_data(
    monkeypatch, sample_annotation_2d_record, sample_record, sample_data_record, calibrated_sensor_record, sensor_record
):
    d = object.__new__(DARTS)
    sd = sample_data_record()
    s = sample_record()
    cs = calibrated_sensor_record()
    ann2d = sample_annotation_2d_record()

    d._sample_data = SimpleNamespace(all=lambda: [sd], get=lambda token: sd)
    d._sample = SimpleNamespace(all=lambda: [s], get=lambda token: s)
    d._calibrated_sensor = SimpleNamespace(get=lambda token: cs)
    d._sensor = SimpleNamespace(get=lambda token: sensor_record())
    d._sample_annotation_2d = SimpleNamespace(all=lambda: [ann2d])

    monkeypatch.setattr(d, "_progress", lambda iterable, *_: iterable)

    DARTS._add_data_to_sample(d)
    DARTS._add_annotations_to_sample_data(d)
    assert sd.anns == (ann2d.token,)


def test_create_relationships(monkeypatch):
    d = object.__new__(DARTS)
    calls = []

    for method in [
        "_add_channel_and_modality_to_sample_data",
        "_add_scene_metadata_to_scene",
        "_add_ins_to_sample",
        "_add_data_to_sample",
        "_add_annotations_to_sample",
        "_add_annotations_to_sample_data",
    ]:
        monkeypatch.setattr(DARTS, method, lambda self, m=method: calls.append(m))

    DARTS._create_relationships(d)
    assert set(calls) == {
        "_add_channel_and_modality_to_sample_data",
        "_add_scene_metadata_to_scene",
        "_add_ins_to_sample",
        "_add_data_to_sample",
        "_add_annotations_to_sample",
        "_add_annotations_to_sample_data",
    }
