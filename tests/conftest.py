from types import SimpleNamespace
from darts import DARTS
import pytest


@pytest.fixture
def sample_data_record():
    def _factory(
        token="sd1",
        sample_token="sample1",
        cal_token="cs1",
        channel=None,
        key_frame=True,
        anns=None,
        prev="",
        next="",
        ego_pose_token="ep1",
        modality=None,
        filename="",
        width=None,
        height=None,
    ):
        return SimpleNamespace(
            token=token,
            sample_token=sample_token,
            calibrated_sensor_token=cal_token,
            ego_pose_token=ego_pose_token,
            channel=channel,
            modality=modality,
            filename=filename,
            is_key_frame=key_frame,
            anns=anns,
            prev=prev,
            next=next,
            width=width,
            height=height,
        )

    return _factory


@pytest.fixture
def calibrated_sensor_record():
    def _factory(token="cs1", sensor_token="sensor1", translation=None, rotation=None, camera_intrinsic=None):
        return SimpleNamespace(
            token=token,
            sensor_token=sensor_token,
            translation=translation,
            rotation=rotation,
            camera_intrinsic=camera_intrinsic,
        )

    return _factory


@pytest.fixture
def ego_pose_record():
    def _factory(token="ep1", translation=None, rotation=None):
        return SimpleNamespace(token=token, translation=translation, rotation=rotation)

    return _factory


@pytest.fixture
def instance_record():
    def _factory(token="i1", category_token=""):
        return SimpleNamespace(token=token, category_token=category_token)

    return _factory


@pytest.fixture
def instance_2d_record():
    def _factory(token="i2d1", category_token=""):
        return SimpleNamespace(token=token, category_token=category_token)

    return _factory


@pytest.fixture
def sensor_record():
    def _factory(token="sensor1", channel="CAM_FRONT", modality="camera"):
        return SimpleNamespace(token=token, channel=channel, modality=modality)

    return _factory


@pytest.fixture
def sample_record():
    def _factory(token="sample1", ins_token=None, anns=None, data=None, prev="", next=""):
        return SimpleNamespace(token=token, data=data, anns=anns, ins_token=ins_token, prev=prev, next=next)

    return _factory


@pytest.fixture
def ins_record():
    def _factory(token="ins1", sample_token="sample1", is_key_frame=True, prev="", next=""):
        return SimpleNamespace(token=token, sample_token=sample_token, is_key_frame=is_key_frame, prev=prev, next=next)

    return _factory


@pytest.fixture
def scene_metadata_record():
    def _factory(token="meta1", scene_token="scene1"):
        return SimpleNamespace(token=token, scene_token=scene_token)

    return _factory


@pytest.fixture
def scene_record():
    def _factory(token="scene1", scene_metadata_token=None, first_sample_token="sample1"):
        return SimpleNamespace(
            token=token, scene_metadata_token=scene_metadata_token, first_sample_token=first_sample_token
        )

    return _factory


@pytest.fixture
def sample_annotation_record():
    def _factory(
        token="ann1",
        sample_token="sample1",
        instance_token="instance1",
        calibrated_sensor_token="",
        size=None,
        translation=None,
        rotation=None,
        num_lidar_pts=10,
        score=1,
    ):
        return SimpleNamespace(
            token=token,
            sample_token=sample_token,
            instance_token=instance_token,
            calibrated_sensor_token=calibrated_sensor_token,
            translation=translation,
            size=size,
            rotation=rotation,
            num_lidar_pts=num_lidar_pts,
            score=score,
        )

    return _factory


@pytest.fixture
def sample_annotation_2d_record():
    def _factory(
        token="ann2d1",
        sample_token="sample1",
        calibrated_sensor_token="cs1",
        instance_token="instance1",
        instance_2d_token="instance2d1",
        corners=None,
    ):
        return SimpleNamespace(
            token=token,
            instance_token=instance_token,
            sample_token=sample_token,
            calibrated_sensor_token=calibrated_sensor_token,
            instance_2d_token=instance_2d_token,
            corners=corners,
        )

    return _factory


@pytest.fixture
def category_record():
    def _factory(token="cat1", name="name"):
        return SimpleNamespace(token=token, name=name)

    return _factory


@pytest.fixture
def metadata_records():
    return [
        SimpleNamespace(
            token="metadata_1",
            scene_token="scene_1",
            int_value=1,
            string_value="1",
            string_list_value=("1", "2"),
            int_list_value=(1, 2),
        ),
        SimpleNamespace(
            token="metadata_2",
            scene_token="scene_2",
            int_value=2,
            string_value="2",
            string_list_value=("2", "3"),
            int_list_value=(2, 3),
        ),
    ]


@pytest.fixture
def darts_dataset(
    sample_record,
    sample_data_record,
    scene_record,
    metadata_records,
    sample_annotation_record,
    sample_annotation_2d_record,
    calibrated_sensor_record,
    ego_pose_record,
    instance_record,
    instance_2d_record,
    ins_record,
):
    d = object.__new__(DARTS)

    s1 = scene_record(token="scene_1", scene_metadata_token="metadata_1", first_sample_token="sample11")
    s2 = scene_record(token="scene_2", scene_metadata_token="metadata_2", first_sample_token="sample21")

    sample11 = sample_record(
        token="sample11", anns=("ann11",), data={"CAM": "sd11"}, ins_token="ins11", prev="", next="sample12"
    )
    sample12 = sample_record(
        token="sample12", anns=("ann12",), data={"CAM": "sd12"}, ins_token="ins12", prev="sample11", next=""
    )

    sample21 = sample_record(
        token="sample21", anns=(), data={"CAM": "sd21"}, ins_token="ins21", prev="", next="sample22"
    )
    sample22 = sample_record(
        token="sample22", anns=(), data={"CAM": "sd23"}, ins_token="ins22", prev="sample21", next=""
    )

    sd11 = sample_data_record(
        token="sd11",
        sample_token="sample11",
        cal_token="cs1",
        channel="CAM",
        modality="camera",
        key_frame=True,
        anns=("ann2d11",),
        prev="",
        next="",
        ego_pose_token="ep11",
    )
    sd12 = sample_data_record(
        token="sd12",
        sample_token="sample12",
        cal_token="cs1",
        channel="CAM",
        modality="camera",
        key_frame=True,
        anns=("ann2d12",),
        prev="",
        next="sd13",
        ego_pose_token="ep12",
    )
    sd13 = sample_data_record(
        token="sd13",
        sample_token="sample12",
        cal_token="cs1",
        channel="CAM",
        modality="camera",
        key_frame=False,
        anns=(),
        prev="sd12",
        next="",
        ego_pose_token="ep13",
    )

    sd21 = sample_data_record(
        token="sd21",
        sample_token="sample21",
        cal_token="cs2",
        channel="CAM",
        modality="camera",
        key_frame=True,
        anns=(),
        prev="",
        next="sd22",
        ego_pose_token="ep21",
    )
    sd22 = sample_data_record(
        token="sd22",
        sample_token="sample21",
        cal_token="cs2",
        channel="CAM",
        modality="camera",
        key_frame=True,
        anns=(),
        prev="sd21",
        next="",
        ego_pose_token="ep22",
    )
    sd23 = sample_data_record(
        token="sd23",
        sample_token="sample22",
        cal_token="cs2",
        channel="CAM",
        modality="camera",
        key_frame=False,
        anns=(),
        prev="",
        next="",
        ego_pose_token="ep23",
    )

    ann11 = sample_annotation_record(token="ann11", sample_token="sample11", instance_token="instance11")
    ann12 = sample_annotation_record(token="ann12", sample_token="sample12", instance_token="instance11")

    ann2d11 = sample_annotation_2d_record(
        token="ann2d11",
        sample_token="sample11",
        calibrated_sensor_token="cs1",
        instance_token="instance11",
        instance_2d_token="instance2d1",
    )
    ann2d12 = sample_annotation_2d_record(
        token="ann2d12",
        sample_token="sample12",
        calibrated_sensor_token="cs1",
        instance_token="instance12",
        instance_2d_token="instance2d2",
    )

    cs1 = calibrated_sensor_record(token="cs1")
    cs2 = calibrated_sensor_record(token="cs2")

    ep11 = ego_pose_record(token="ep11")
    ep12 = ego_pose_record(token="ep12")
    ep13 = ego_pose_record(token="ep13")
    ep21 = ego_pose_record(token="ep21")
    ep22 = ego_pose_record(token="ep22")
    ep23 = ego_pose_record(token="ep23")

    i11 = instance_record(token="instance11")

    i2d1 = instance_2d_record(token="instance2d1")
    i2d2 = instance_2d_record(token="instance2d2")

    ins11 = ins_record(token="ins11", sample_token="sample11", prev="", next="ins12")
    ins12 = ins_record(token="ins12", sample_token="sample12", prev="ins11", next="")
    ins21 = ins_record(token="ins21", sample_token="sample21", prev="", next="ins22")
    ins22 = ins_record(token="ins22", sample_token="sample22", prev="ins21", next="")

    def rc(items):
        return SimpleNamespace(
            all=lambda: items,
            get=lambda token: next(x for x in items if x.token == token),
        )

    d._scene_metadata = rc(metadata_records)
    d._scene = rc([s1, s2])
    d._sample = rc([sample11, sample12, sample21, sample22])
    d._sample_data = rc([sd11, sd12, sd13, sd21, sd22, sd23])
    d._sample_annotation = rc([ann11, ann12])
    d._sample_annotation_2d = rc([ann2d11, ann2d12])
    d._calibrated_sensor = rc([cs1, cs2])
    d._ego_pose = rc([ep11, ep12, ep13, ep21, ep22, ep23])
    d._instance = rc([i11])
    d._instance_2d = rc([i2d1, i2d2])
    d._ins = rc([ins11, ins12, ins21, ins22])

    d._category = rc([])
    d._sensor = rc([])

    return d
