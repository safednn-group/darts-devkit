import pytest
from types import SimpleNamespace

from darts_devkit import DARTS
from darts_devkit.visualization.rerun_visualizer import RerunVisualizer
import numpy as np


@pytest.fixture
def mock_rr(monkeypatch):
    calls = []

    monkeypatch.setattr("rerun.log", lambda *a, **k: calls.append(("log", a, k)))
    monkeypatch.setattr("rerun.init", lambda *a, **k: calls.append(("init", a, k)))
    monkeypatch.setattr("rerun.spawn", lambda *a, **k: calls.append(("spawn", a, k)))
    monkeypatch.setattr("rerun.set_time", lambda *a, **k: calls.append(("set_time", a, k)))

    return calls


def recorder(name, calls):
    return lambda _darts, *args: calls.append((name, *args))


def test_get_keyframe_sample_data(sample_data_record, sample_record):
    d = object.__new__(DARTS)
    viz = RerunVisualizer()

    sample = sample_record(data={"CAM_FRONT": "cam_front_token", "LIDAR_TOP": "lidar_top_token"})
    sd_cam = sample_data_record(token="cam_front_token", modality="camera")
    sd_lidar = sample_data_record(token="lidar_top_token", modality="lidar")

    sample_data_dict = {
        sd_cam.token: sd_cam,
        sd_lidar.token: sd_lidar,
    }

    d._sample_data = SimpleNamespace(get=lambda token: sample_data_dict[token])

    result = viz._get_keyframe_sample_data(d, sample, "camera")

    assert len(result) == 1
    assert result[0].token == "cam_front_token"
    assert result[0].modality == "camera"


def test_get_keyframe_sample_data_empty(sample_data_record, sample_record):
    d = object.__new__(DARTS)
    viz = RerunVisualizer()

    sample = sample_record(data={"LIDAR_TOP": "lidar_token"})
    sd = sample_data_record(token="lidar_token", modality="lidar")

    d._sample_data = SimpleNamespace(get=lambda token: sd)

    result = viz._get_keyframe_sample_data(d, sample, "camera")

    assert result == []


def test_get_lidar_pointclouds(sample_data_record):
    viz = RerunVisualizer()

    sd1 = sample_data_record(token="t1")
    sd2 = sample_data_record(token="t2")
    sample_datas = [sd1, sd2]

    pc1 = SimpleNamespace(points="pc1")
    pc2 = SimpleNamespace(points="pc2")

    calls = []

    class FakeDarts:
        def get_lidar_pointcloud(self, sd):
            calls.append(sd.token)
            return {"t1": pc1, "t2": pc2}[sd.token]

    result = viz._get_lidar_pointclouds(FakeDarts(), sample_datas)

    assert result == {"t1": pc1, "t2": pc2}
    assert calls == ["t1", "t2"]


def test_visualize_iterates_samples(monkeypatch, darts_dataset, mock_rr):
    viz = RerunVisualizer()
    d = darts_dataset

    calls = []

    monkeypatch.setattr(viz, "_log_frame", lambda darts, sample: calls.append(sample.token))

    viz.visualize(d, "scene_1")
    assert calls == ["sample11", "sample12"]
    assert any(c[0] == "init" for c in mock_rr)
    assert any(c[0] == "spawn" for c in mock_rr)
    set_time_calls = [c for c in mock_rr if c[0] == "set_time"]
    sequences = [c[2]["sequence"] for c in set_time_calls]

    assert sequences == [1, 2]


def test_log_frame_flow(monkeypatch, sample_record):
    viz = RerunVisualizer()
    sample = sample_record(token="s1")

    annotations = ["ann"]
    lidar_sds = ["lidar_sd"]
    lidar_pcs = {"lidar_sd": "pc"}
    camera_sds = ["cam_sd"]
    annotations_2d = ["ann2d"]

    class FakeDarts:
        def get_annotations_from_samples(self, samples):
            assert samples == [sample]
            return annotations

        def get_annotations_2d_from_sample_datas(self, sds):
            assert sds == camera_sds
            return annotations_2d

    calls = []
    modalities = []

    def fake_get_keyframe(d, s, m):
        modalities.append(m)
        return lidar_sds if m == "lidar" else camera_sds

    monkeypatch.setattr(viz, "_log_annotations", recorder("log_annotations", calls))
    monkeypatch.setattr(viz, "_get_keyframe_sample_data", fake_get_keyframe)
    monkeypatch.setattr(viz, "_get_lidar_pointclouds", lambda d, sds: lidar_pcs)
    monkeypatch.setattr(viz, "_log_lidar_pointclouds", recorder("log_lidar", calls))
    monkeypatch.setattr(viz, "_log_camera_images", recorder("log_camera", calls))
    monkeypatch.setattr(viz, "_log_annotations_2d", recorder("log_ann2d", calls))

    viz._log_frame(FakeDarts(), sample)
    assert modalities == ["lidar", "camera"]
    assert calls[0] == ("log_annotations", annotations)
    assert calls[1] == ("log_lidar", lidar_sds, lidar_pcs)
    assert calls[2] == ("log_camera", camera_sds)
    assert calls[3] == ("log_ann2d", annotations_2d)


def test_log_annotations_2d(monkeypatch, sample_annotation_2d_record, sensor_record, category_record):
    viz = RerunVisualizer()

    calls = []

    monkeypatch.setattr("rerun.log", lambda *a, **k: calls.append((a, k)))

    ann1 = sample_annotation_2d_record(token="t1", corners=(0, 0, 1, 1))
    ann2 = sample_annotation_2d_record(token="t2", corners=(1, 1, 2, 2))

    class FakeDarts:
        def get_sensor_from_annotation_2d(self, token):
            return sensor_record(channel="CAM_FRONT" if token == "t1" else "CAM_BACK")

        def get_category_from_annotation_2d(self, token):
            return category_record(name=f"label_{token}")

    viz._log_annotations_2d(FakeDarts(), [ann1, ann2])

    assert len(calls) == 2

    paths = [c[0][0] for c in calls]
    assert "world/ego_vehicle/CAM_FRONT/annotations" in paths
    assert "world/ego_vehicle/CAM_BACK/annotations" in paths


def test_log_camera_images(monkeypatch, sample_data_record):
    viz = RerunVisualizer()

    calls = []

    monkeypatch.setattr(viz, "_log_sensor_calibration", lambda d, sd: calls.append(("calib", sd.channel)))
    monkeypatch.setattr(viz, "_log_camera", lambda img, ch: calls.append(("camera", ch)))

    sd1 = sample_data_record(channel="CAM_FRONT")
    sd2 = sample_data_record(channel="CAM_BACK")

    class FakeDarts:
        def get_image(self, sd):
            return "image"

    viz._log_camera_images(FakeDarts(), [sd1, sd2])

    assert calls == [
        ("calib", "CAM_FRONT"),
        ("camera", "CAM_FRONT"),
        ("calib", "CAM_BACK"),
        ("camera", "CAM_BACK"),
    ]


def test_log_camera(monkeypatch):
    viz = RerunVisualizer()

    calls = []

    monkeypatch.setattr("rerun.log", lambda *a, **k: calls.append((a, k)))

    class FakeImage:
        format = "PNG"

        def save(self, *args, **kwargs):
            pass

        def __array__(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

    viz._log_camera(FakeImage(), "CAM_FRONT")

    assert len(calls) == 1
    assert calls[0][0][0] == "world/ego_vehicle/CAM_FRONT"


def test_log_lidar_pointclouds(monkeypatch, sample_data_record):
    viz = RerunVisualizer()

    calls = []

    monkeypatch.setattr(viz, "_log_sensor_calibration", lambda d, sd: calls.append(("calib", sd.token)))
    monkeypatch.setattr(viz, "_log_ego_pose", lambda d, sd: calls.append(("ego", sd.token)))
    monkeypatch.setattr(viz, "_log_lidar", lambda pc, ch, max_intensity=None: calls.append(("lidar", ch)))

    sd1 = sample_data_record(token="t1", channel="LIDAR_TOP")
    sd2 = sample_data_record(token="t2", channel="LIDAR_FRONT")

    pc = np.zeros((4, 4))
    lidar_pointclouds = {
        "t1": SimpleNamespace(points=pc),
        "t2": SimpleNamespace(points=pc),
    }

    viz._log_lidar_pointclouds(object(), [sd1, sd2], lidar_pointclouds)

    assert calls == [
        ("calib", "t1"),
        ("ego", "t1"),
        ("lidar", "LIDAR_TOP"),
        ("calib", "t2"),
        ("ego", "t2"),
        ("lidar", "LIDAR_FRONT"),
    ]


def test_log_lidar(monkeypatch):
    viz = RerunVisualizer()

    calls = []

    monkeypatch.setattr("rerun.log", lambda *a, **k: calls.append((a, k)))

    pointcloud = np.array(
        [
            [1, 2, 3, 100],
            [4, 5, 6, 200],
        ],
        dtype=float,
    )

    viz._log_lidar(pointcloud, "LIDAR_TOP", max_intensity=255)

    assert len(calls) == 1
    assert calls[0][0][0] == "world/ego_vehicle/LIDAR_TOP"


def test_log_ego_pose(monkeypatch, ego_pose_record, sample_data_record):
    viz = RerunVisualizer()

    calls = []

    monkeypatch.setattr("rerun.log", lambda *a, **k: calls.append((a, k)))

    ego_pose = ego_pose_record(
        translation=[1, 2, 3],
        rotation=[0, 0, 0, 1],
    )

    class FakeDarts:
        ego_pose = SimpleNamespace(get=lambda token: ego_pose)

    sd = sample_data_record(ego_pose_token="pose1")

    viz._log_ego_pose(FakeDarts(), sd)

    assert len(calls) == 1
    assert calls[0][0][0] == "world/ego_vehicle"


def test_log_sensor_calibration_camera(monkeypatch, calibrated_sensor_record, sample_data_record):
    viz = RerunVisualizer()

    calls = []

    monkeypatch.setattr("rerun.log", lambda *a, **k: calls.append((a, k)))

    calib = calibrated_sensor_record(
        translation=[0, 0, 0],
        rotation=[0, 0, 0, 1],
        camera_intrinsic=np.eye(3),
    )

    class FakeDarts:
        calibrated_sensor = SimpleNamespace(get=lambda token: calib)

    sd = sample_data_record(
        cal_token="c1",
        channel="CAM_FRONT",
        width=800,
        height=600,
    )

    viz._log_sensor_calibration(FakeDarts(), sd)

    assert len(calls) == 2  # transform + pinhole


def test_log_sensor_calibration_lidar(monkeypatch, calibrated_sensor_record, sample_data_record):
    viz = RerunVisualizer()

    calls = []

    monkeypatch.setattr("rerun.log", lambda *a, **k: calls.append((a, k)))

    calib = calibrated_sensor_record(
        translation=[0, 0, 0],
        rotation=[0, 0, 0, 1],
    )

    class FakeDarts:
        calibrated_sensor = SimpleNamespace(get=lambda token: calib)

    sd = sample_data_record(
        cal_token="c1",
        channel="LIDAR_TOP",
    )

    viz._log_sensor_calibration(FakeDarts(), sd)

    assert len(calls) == 1  # only transform


def test_log_annotations(monkeypatch, sample_annotation_record):
    viz = RerunVisualizer()

    calls = []

    monkeypatch.setattr("rerun.log", lambda *a, **k: calls.append((a, k)))

    ann = sample_annotation_record(
        token="a1",
        size=[1, 2, 3],
        translation=[0, 0, 0],
        rotation=[0, 0, 0, 1],
    )

    class FakeDarts:
        def get_category_from_annotation(self, token):
            return SimpleNamespace(name="car")

    viz._log_annotations(FakeDarts(), [ann])

    assert len(calls) == 1
    assert calls[0][0][0] == "world/annotations"
