from types import SimpleNamespace

import pytest

from darts.dataset.darts import DARTS


@pytest.mark.parametrize(
    "query",
    [{"int_value": "1"}, {"string_value": 1}, {"int_value": [1, "2"]}],
)
def test_match_query_type_mismatch(metadata_records, query):
    d = object.__new__(DARTS)

    with pytest.raises(TypeError):
        d._match_query(metadata_records[0], query)


@pytest.mark.parametrize(
    "query, expected",
    [
        ({"int_value": 1}, [True, False]),
        ({"int_value": [1, 2]}, [True, True]),
        ({"string_value": "1"}, [True, False]),
        ({"string_value": ["1", "2"]}, [True, True]),
        ({"string_list_value": ["1"]}, [True, False]),
        ({"string_list_value": ["1", "2"]}, [True, True]),
        ({"int_list_value": [1]}, [True, False]),
        ({"int_list_value": [1, 2]}, [True, True]),
    ],
)
def test_match_query_simple(metadata_records, query, expected):
    d = object.__new__(DARTS)

    results = [d._match_query(record, query) for record in metadata_records]

    assert results == expected


@pytest.mark.parametrize(
    "query, expected",
    [
        ({"not": {"int_value": 1}}, [False, True]),
        ({"and": [{"int_value": 1}, {"string_list_value": "3"}]}, [False, False]),
        ({"or": [{"int_value": 1}, {"string_list_value": "3"}]}, [True, True]),
        ({"not": {"or": [{"int_value": 1}, {"string_list_value": "3"}]}}, [False, False]),
    ],
)
def test_match_query_with_operands(metadata_records, query, expected):
    d = object.__new__(DARTS)

    results = [d._match_query(record, query) for record in metadata_records]

    assert results == expected


@pytest.mark.parametrize("query", [({"and": ["not-a-list"]}), ({"a"}), (1), {"and": {"ad"}}])
def test_match_query_invalid_structure(metadata_records, query):
    d = object.__new__(DARTS)

    with pytest.raises(Exception):
        d._match_query(metadata_records[0], query)


def test_ins_from_samples(sample_record, ins_record):
    d = object.__new__(DARTS)
    s = sample_record(ins_token="ins2")
    i1 = ins_record(token="ins1", is_key_frame=False, prev="", next="ins2")
    i2 = ins_record(token="ins2", is_key_frame=True, prev="ins1", next="ins3")
    i3 = ins_record(token="ins3", is_key_frame=False, prev="ins2", next="")
    ins_map = {i1.token: i1, i2.token: i2, i3.token: i3}

    d._sample = SimpleNamespace(all=lambda: [s], get=lambda token: s)
    d._ins = SimpleNamespace(all=lambda: ins_map.values(), get=lambda token: ins_map[token])

    assert list(ins_map.values()) == DARTS._ins_from_samples(d, [s])


def test_annotations_from_sample_datas(sample_data_record, sample_annotation_2d_record):
    d = object.__new__(DARTS)
    ann2d1 = sample_annotation_2d_record(token="ann2d1")
    ann2d2 = sample_annotation_2d_record(token="ann2d2")
    ann2d3 = sample_annotation_2d_record(token="ann2d3")
    anns_2d_map = {ann2d1.token: ann2d1, ann2d2.token: ann2d2, ann2d3.token: ann2d3}
    sd = sample_data_record(anns=list(anns_2d_map.keys()))
    d._sample_annotation_2d = SimpleNamespace(get=lambda token: anns_2d_map[token])
    assert list(anns_2d_map.values()) == DARTS._annotations_from_sample_datas(d, [sd])


def test_annotations_from_samples(sample_record, sample_annotation_record):
    d = object.__new__(DARTS)
    ann1 = sample_annotation_record(token="ann2d1")
    ann2 = sample_annotation_record(token="ann2d2")
    ann3 = sample_annotation_record(token="ann2d3")
    anns_map = {ann1.token: ann1, ann2.token: ann2, ann3.token: ann3}
    s = sample_record(anns=list(anns_map.keys()))
    d._sample_annotation = SimpleNamespace(get=lambda token: anns_map[token])
    assert list(anns_map.values()) == DARTS._annotations_from_samples(d, [s])


def test_sample_data_from_samples(sample_record, sample_data_record):
    d = object.__new__(DARTS)
    sd1 = sample_data_record(token="sd1", key_frame=False, prev="", next="sd2")
    sd2 = sample_data_record(token="sd2", key_frame=True, prev="sd1", next="sd3")
    sd3 = sample_data_record(token="sd3", key_frame=False, prev="sd2", next="")
    sd_map = {sd1.token: sd1, sd2.token: sd2, sd3.token: sd3}
    s = sample_record(data={"CAM_FRONT": sd2.token})

    d._sample = SimpleNamespace(all=lambda: [s], get=lambda token: s)
    d._sample_data = SimpleNamespace(all=lambda: sd_map.values(), get=lambda token: sd_map[token])

    assert list(sd_map.values()) == DARTS._sample_data_from_samples(d, [s])


def test_samples_from_scenes(scene_record, sample_record):
    d = object.__new__(DARTS)
    scene = scene_record(token="scene1", first_sample_token="sample1")
    sample1 = sample_record(token="sample1", prev="", next="sample2")
    sample2 = sample_record(token="sample2", prev="sample1", next="sample3")
    sample3 = sample_record(token="sample3", prev="sample2", next="")
    sample_map = {sample1.token: sample1, sample2.token: sample2, sample3.token: sample3}

    d._sample = SimpleNamespace(all=lambda: list(sample_map.values()), get=lambda token: sample_map[token])
    d._scene = SimpleNamespace(get=lambda token: scene)
    assert list(sample_map.values()) == DARTS._samples_from_scenes(d, [scene])


@pytest.mark.parametrize(
    "query",
    [
        ({"int_value": 1}),
        ({"not": {"and": [{"int_value": 2}, {"string_list_value": "3"}]}}),
        ({"or": [{"int_value": 1}, {"string_list_value": ["4", "5"]}]}),
        ({"not": {"not": {"or": [{"int_list_value": 1}, {"int_list_value": 10}]}}}),
    ],
)
def test_filter_scenes_first(darts_dataset, query):
    filtered = darts_dataset.filter_scenes(query)

    scenes = filtered.scene.all()
    assert scenes[0].token == "scene_1"
    assert len(filtered.scene_metadata.all()) == 1
    assert len(scenes) == 1
    assert len(filtered.sample.all()) == 2
    assert len(filtered.sample_data.all()) == 3
    assert len(filtered.sample_annotation.all()) == 2
    assert len(filtered.sample_annotation_2d.all()) == 2
    assert len(filtered.ins.all()) == 2
    assert len(filtered.instance.all()) == 1
    assert len(filtered.instance_2d.all()) == 2
    assert len(filtered.ego_pose.all()) == 3
    assert len(filtered.calibrated_sensor.all()) == 1


@pytest.mark.parametrize(
    "query",
    [
        ({"not": {"int_value": 1}}),
        ({"not": {"not": {"and": [{"int_value": 2}, {"string_list_value": "3"}]}}}),
        ({"not": {"or": [{"int_value": 1}, {"string_list_value": ["4", "5"]}]}}),
        ({"not": {"not": {"not": {"or": [{"int_list_value": 1}, {"int_list_value": 10}]}}}}),
    ],
)
def test_filter_scenes_second(darts_dataset, query):
    filtered = darts_dataset.filter_scenes(query)

    scenes = filtered.scene.all()
    assert scenes[0].token == "scene_2"
    assert len(filtered.scene_metadata.all()) == 1
    assert len(scenes) == 1
    assert len(filtered.sample.all()) == 2
    assert len(filtered.sample_data.all()) == 3
    assert len(filtered.sample_annotation.all()) == 0
    assert len(filtered.sample_annotation_2d.all()) == 0
    assert len(filtered.ins.all()) == 2
    assert len(filtered.instance.all()) == 0
    assert len(filtered.instance_2d.all()) == 0
    assert len(filtered.ego_pose.all()) == 3
    assert len(filtered.calibrated_sensor.all()) == 1


def test_filter_scenes_no_results(darts_dataset):
    filtered = darts_dataset.filter_scenes({"int_value": 999})

    assert len(filtered.scene_metadata.all()) == 0
    assert len(filtered.scene.all()) == 0
    assert len(filtered.sample.all()) == 0
    assert len(filtered.sample_data.all()) == 0
    assert len(filtered.sample_annotation.all()) == 0
    assert len(filtered.sample_annotation_2d.all()) == 0
    assert len(filtered.ins.all()) == 0
    assert len(filtered.instance.all()) == 0
    assert len(filtered.instance_2d.all()) == 0
    assert len(filtered.ego_pose.all()) == 0
    assert len(filtered.calibrated_sensor.all()) == 0


def test_filter_scenes_multiple_results(darts_dataset):
    filtered = darts_dataset.filter_scenes({"int_value": [1, 2]})

    assert len(filtered.scene_metadata.all()) == 2
    assert len(filtered.scene.all()) == 2
    assert len(filtered.sample.all()) == 4
    assert len(filtered.sample_data.all()) == 6
    assert len(filtered.sample_annotation.all()) == 2
    assert len(filtered.sample_annotation_2d.all()) == 2
    assert len(filtered.ins.all()) == 4
    assert len(filtered.instance.all()) == 1
    assert len(filtered.instance_2d.all()) == 2
    assert len(filtered.ego_pose.all()) == 6
    assert len(filtered.calibrated_sensor.all()) == 2
