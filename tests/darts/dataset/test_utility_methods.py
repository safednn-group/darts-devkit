import pytest
import numpy as np
from pathlib import Path
from PIL import Image

from darts_devkit import DARTS
from darts_devkit.dataset.data_classes import LidarPointCloud


def test_get_lidar_pointcloud_wrong_modality(sample_data_record):
    d = object.__new__(DARTS)
    sd = sample_data_record(modality="camera")
    with pytest.raises(TypeError):
        d.get_lidar_pointcloud(sd)


def test_get_lidar_pointcloud_wrong_filetype(monkeypatch, sample_data_record):
    monkeypatch.setattr(DARTS, "_load_table", lambda self, name, cls: None)
    monkeypatch.setattr(DARTS, "_create_relationships", lambda self: None)
    monkeypatch.setattr(DARTS, "_load_splits_table", lambda self: None)

    d = DARTS("data", "v1")
    sd = sample_data_record(modality="lidar", filename="filename.png")
    with pytest.raises(TypeError):
        d.get_lidar_pointcloud(sd)


def test_get_lidar_pointcloud(tmp_path, monkeypatch, sample_data_record):
    file_path = tmp_path / "test.bin"
    data = np.array(
        [
            [1.0, 2.0, 3.0, 0.5, 1],
            [4.0, 5.0, 6.0, 0.7, 2],
        ],
        dtype=np.float32,
    )

    data.tofile(file_path)
    monkeypatch.setattr(DARTS, "_load_table", lambda self, name, cls: None)
    monkeypatch.setattr(DARTS, "_create_relationships", lambda self: None)
    monkeypatch.setattr(DARTS, "_load_splits_table", lambda self: None)

    d = DARTS("data", "v1")
    monkeypatch.setattr(DARTS, "_get_sensor_data_path", lambda self, sd: Path(file_path))
    sd = sample_data_record(modality="lidar", filename="test.bin")
    pc = d.get_lidar_pointcloud(sd)

    assert isinstance(pc, LidarPointCloud)
    assert pc.points.shape[0] == LidarPointCloud.nbr_dims()
    assert pc.points.shape[1] == pc.nbr_points()


def test_get_image(tmp_path, monkeypatch, sample_data_record):
    file_path = tmp_path / "test.png"
    img = Image.new("RGB", (10, 10), color="red")
    img.save(file_path)
    monkeypatch.setattr(DARTS, "_load_table", lambda self, name, cls: None)
    monkeypatch.setattr(DARTS, "_create_relationships", lambda self: None)
    monkeypatch.setattr(DARTS, "_load_splits_table", lambda self: None)

    d = DARTS("data", "v1")
    monkeypatch.setattr(DARTS, "_get_sensor_data_path", lambda self, sd: Path(file_path))
    sd = sample_data_record(modality="camera", filename="test.png")
    result = d.get_image(sd)
    assert result is not None
    assert result.size == (10, 10)
    assert result.mode == "RGB"


def test_get_sensor_from_annotation_2d(sample_annotation_2d_record, calibrated_sensor_record, sensor_record):
    d = object.__new__(DARTS)
    annotation_2d = sample_annotation_2d_record(calibrated_sensor_token="calib_token")
    calibrated_sensor = calibrated_sensor_record(sensor_token="sensor_token")
    sensor = sensor_record(channel="camera_front")
    d._sample_annotation_2d = {"ann_token": annotation_2d}
    d._calibrated_sensor = {"calib_token": calibrated_sensor}
    d._sensor = {"sensor_token": sensor}
    result = d.get_sensor_from_annotation_2d("ann_token")
    assert result is sensor
    assert result.channel == "camera_front"


def test_get_category_from_annotation_2d(sample_annotation_2d_record, instance_2d_record, category_record):
    d = object.__new__(DARTS)
    annotation_2d = sample_annotation_2d_record(token="annotation_2d_token", instance_2d_token="instance_token")
    instance_2d = instance_2d_record(token="instance_token", category_token="category_token")
    category = category_record(token="category_token", name="name")
    d._sample_annotation_2d = {"annotation_2d_token": annotation_2d}
    d._instance_2d = {"instance_token": instance_2d}
    d._category = {"category_token": category}
    result = d.get_category_from_annotation_2d("annotation_2d_token")
    assert result is category
    assert result.name == "name"


def test_get_category_from_annotation(sample_annotation_record, instance_record, category_record):
    d = object.__new__(DARTS)
    annotation = sample_annotation_record(token="annotation_token", instance_token="instance_token")
    instance = instance_record(token="instance_token", category_token="category_token")
    category = category_record(token="category_token", name="name")
    d._sample_annotation = {"annotation_token": annotation}
    d._instance = {"instance_token": instance}
    d._category = {"category_token": category}
    result = d.get_category_from_annotation("annotation_token")
    assert result is category
    assert result.name == "name"
