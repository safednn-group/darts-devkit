import math

import pytest
from pyquaternion import Quaternion

from darts.evaluation.waymo_evaluator import WaymoEvaluator
from darts.dataset.user_models import Box


def make_box(center, size, yaw=0.0):
    q = Quaternion(axis=[0, 0, 1], angle=yaw)

    return Box(center=center, size=size, orientation=[q.w, q.x, q.y, q.z], track_id=1, score=1, name="box")


@pytest.mark.parametrize(
    ("b1", "b2", "expected"),
    [
        pytest.param(
            make_box([0, 0, 0], [2, 2, 2]),
            make_box([0, 0, 0], [2, 2, 2]),
            1.0,
            id="identical_boxes",
        ),
        pytest.param(
            make_box([0, 0, 0], [2, 2, 2]),
            make_box([2.01, 0, 0], [2, 2, 2]),
            0.0,
            id="no_xy_overlap",
        ),
        pytest.param(
            make_box([0, 0, 0], [2, 2, 2]),
            make_box([0, 0, 2.01], [2, 2, 2]),
            0.0,
            id="no_z_overlap",
        ),
        pytest.param(
            make_box([0, 0, 0], [2, 2, 2]),
            make_box([1, 0, 0], [2, 2, 2]),
            1 / 3,
            id="partial_overlap_xy",
        ),
        pytest.param(
            make_box([0, 0, 0], [2, 2, 2]),
            make_box([0, 0, 1], [2, 2, 2]),
            1 / 3,
            id="partial_overlap_z",
        ),
        pytest.param(
            make_box([0, 0, 0], [2, 2, 2]),
            make_box([2, 0, 0], [2, 2, 2]),
            0.0,
            id="touching_faces",
        ),
        pytest.param(
            make_box([0, 0, 0], [2, 2, 2]),
            make_box([0, 0, 0], [4, 4, 2]),
            0.25,
            id="one_inside_the_other",
        ),
        pytest.param(
            make_box([0, 0, 0], [2, 2, 2]),
            make_box([2, 2, 2], [2, 2, 2]),
            0.0,
            id="touching_vertex",
        ),
    ],
)
def test_compute_iou(b1, b2, expected):
    evaluator = WaymoEvaluator()

    iou = evaluator._compute_iou(b1, b2)

    assert iou == pytest.approx(expected)
    assert iou == pytest.approx(evaluator._compute_iou(b2, b1))
