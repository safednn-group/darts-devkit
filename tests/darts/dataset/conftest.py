import pytest
from types import SimpleNamespace

@pytest.fixture
def sample_data_record():
    def _factory(token="sd1", cal_token="cs1", channel=None, key_frame=True):
        return SimpleNamespace(
            token=token,
            sample_token="sample1",
            calibrated_sensor_token=cal_token,
            channel=channel,
            is_key_frame=key_frame,
            anns=None,
        )
    return _factory

@pytest.fixture
def calibrated_sensor_record():
    def _factory(token="cs1", sensor_token="sensor1"):
        return SimpleNamespace(token=token, sensor_token=sensor_token)
    return _factory

@pytest.fixture
def sensor_record():
    def _factory(token="sensor1", channel="CAM_FRONT"):
        return SimpleNamespace(token=token, channel=channel)
    return _factory

@pytest.fixture
def sample_record():
    def _factory(token="sample1"):
        return SimpleNamespace(token=token, data=None, anns=None, ins_token=None)
    return _factory

@pytest.fixture
def ins_record():
    def _factory(token="ins1", sample_token="sample1", is_key_frame=True):
        return SimpleNamespace(token=token, sample_token=sample_token, is_key_frame=is_key_frame)
    return _factory

@pytest.fixture
def scene_metadata_record():
    def _factory(token="meta1", scene_token="scene1"):
        return SimpleNamespace(token=token, scene_token=scene_token)
    return _factory

@pytest.fixture
def scene_record():
    def _factory(token="scene1", scene_metadata_token=None):
        return SimpleNamespace(token=token, scene_metadata_token=scene_metadata_token)
    return _factory

@pytest.fixture
def sample_annotation_record():
    def _factory(token="ann1", sample_token="sample1"):
        return SimpleNamespace(token=token, sample_token=sample_token)
    return _factory

@pytest.fixture
def sample_annotation_2d_record():
    def _factory(token="ann2d1", sample_token="sample1", calibrated_sensor_token="cs1"):
        return SimpleNamespace(token=token, sample_token=sample_token, calibrated_sensor_token=calibrated_sensor_token)
    return _factory