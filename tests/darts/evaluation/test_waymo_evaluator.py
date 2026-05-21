import pytest
from pyquaternion import Quaternion
from types import SimpleNamespace
from darts.evaluation.waymo_evaluator import WaymoEvaluator, WaymoEvaluationConfig, ClassThresholdConfig
from darts.dataset.user_models import Box, Frame, DARTSAnnotations
from darts.dataset.darts import DARTS


def make_box(center, size, yaw=0.0, name="box", score=1):
    q = Quaternion(axis=[0, 0, 1], angle=yaw)

    return Box(center=center, size=size, orientation=[q.w, q.x, q.y, q.z], score=score, name=name)


def make_darts(
    scene_record,
    sample_record,
    sample_annotation_record,
    instance_record,
    category_record,
    gt_frames,
):
    d = object.__new__(DARTS)

    first_sample = gt_frames[0]["sample"]

    scene = scene_record(
        token="scene_1",
        first_sample_token=first_sample,
    )

    samples = {}
    sample_annotations = {}
    instances = {}
    categories = {}

    category_names = set()

    for frame_idx, frame in enumerate(gt_frames):
        ann_tokens = []

        next_token = gt_frames[frame_idx + 1]["sample"] if frame_idx + 1 < len(gt_frames) else ""

        for box_idx, box in enumerate(frame["boxes"]):
            ann_token = f"ann_{frame_idx}_{box_idx}"

            ann = sample_annotation_record(
                token=ann_token,
                instance_token=box["instance"],
                size=box.get("size", [1, 1, 1]),
                translation=box["center"],
                rotation=[1, 0, 0, 0],
                num_lidar_pts=box.get("num_lidar_pts", 10),
            )

            sample_annotations[ann_token] = ann
            ann_tokens.append(ann_token)

            category_token = f"cat_{box['name']}"

            instances.setdefault(
                box["instance"],
                instance_record(
                    token=box["instance"],
                    category_token=category_token,
                ),
            )

            if box["name"] not in category_names:
                categories[category_token] = category_record(
                    token=category_token,
                    name=box["name"],
                )
                category_names.add(box["name"])

        samples[frame["sample"]] = sample_record(
            token=frame["sample"],
            next=next_token,
            anns=ann_tokens,
        )

    d._scene = SimpleNamespace(get=lambda token: {scene.token: scene}[token])
    d._sample = SimpleNamespace(get=lambda token: samples[token])
    d._sample_annotation = SimpleNamespace(get=lambda token: sample_annotations[token])
    d._instance = SimpleNamespace(get=lambda token: instances[token])
    d._category = SimpleNamespace(get=lambda token: categories[token])

    return d


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
        pytest.param(
            make_box([0, 0, 0], [1, 1, 1]),
            make_box([0.333333333333333334, 0, 0], [1, 1, 1]),
            0.5,
            id="small_move",
        ),
    ],
)
def test_compute_iou(b1, b2, expected):
    evaluator = WaymoEvaluator()
    p1 = evaluator._box_to_waymo_box(b1)
    p2 = evaluator._box_to_waymo_box(b2)
    assert evaluator._compute_iou(p1, p2) == pytest.approx(expected)
    assert evaluator._compute_iou(p2, p1) == pytest.approx(expected)


@pytest.mark.parametrize(
    "gt_frames,pred_frames,expected_ap",
    [
        pytest.param(
            [
                {
                    "sample": "s1",
                    "boxes": [
                        {
                            "instance": "car_1",
                            "center": [0, 0, 0],
                            "name": "car",
                        },
                        {
                            "instance": "car_2",
                            "center": [10, 0, 0],
                            "name": "car",
                        },
                    ],
                }
            ],
            [
                {
                    "sample": "s1",
                    "boxes": [
                        {
                            "center": [0, 0, 0],
                            "name": "car",
                        },
                        {
                            "center": [10, 0, 0],
                            "name": "car",
                        },
                    ],
                }
            ],
            1.0,
            id="two_gt_two_tp",
        ),
        pytest.param(
            [
                {
                    "sample": "s1",
                    "boxes": [
                        {
                            "instance": "car_1",
                            "center": [0, 0, 0],
                            "name": "car",
                        },
                        {
                            "instance": "car_2",
                            "center": [10, 0, 0],
                            "name": "car",
                        },
                    ],
                }
            ],
            [
                {
                    "sample": "s1",
                    "boxes": [
                        {
                            "center": [0, 0, 0],
                            "name": "car",
                        },
                    ],
                }
            ],
            0.5,
            id="one_missed_detection",
        ),
        pytest.param(
            [
                {
                    "sample": "s1",
                    "boxes": [
                        {
                            "instance": "car_1",
                            "center": [0, 0, 0],
                            "name": "car",
                        },
                        {
                            "instance": "car_2",
                            "center": [10, 0, 0],
                            "name": "car",
                        },
                    ],
                }
            ],
            [
                {
                    "sample": "s1",
                    "boxes": [
                        {
                            "center": [0.34, 0, 0],
                            "name": "car",
                        },
                    ],
                }
            ],
            0.0,
            id="one_missed_one_thhreshold_detection",
        ),
        pytest.param(
            [
                {
                    "sample": "s1",
                    "boxes": [
                        {
                            "instance": "car_1",
                            "center": [0, 0, 0],
                            "name": "car",
                        },
                        {
                            "instance": "pedestrian_1",
                            "center": [10, 0, 0],
                            "name": "pedestrian",
                        },
                    ],
                }
            ],
            [
                {
                    "sample": "s1",
                    "boxes": [
                        {
                            "center": [0, 0, 0],
                            "name": "car",
                        },
                        {
                            "center": [10, 0, 0],
                            "name": "pedestrian",
                        },
                    ],
                }
            ],
            1.0,
            id="two_gt_two_tp_two_classes",
        ),
        pytest.param(
            [
                {
                    "sample": "s1",
                    "boxes": [
                        {
                            "instance": "car_1",
                            "center": [0, 0, 0],
                            "name": "car",
                        },
                        {
                            "instance": "pedestrian_1",
                            "center": [10, 0, 0],
                            "name": "pedestrian",
                        },
                    ],
                }
            ],
            [
                {
                    "sample": "s1",
                    "boxes": [
                        {
                            "center": [0, 0, 0],
                            "name": "car",
                        },
                        {
                            "center": [10, 0, 0],
                            "name": "car",
                        },
                    ],
                }
            ],
            0.25,
            id="two_gt_one_tp_two_classes",
        ),
    ],
)
def test_evaluate(
    scene_record,
    sample_record,
    sample_annotation_record,
    instance_record,
    category_record,
    gt_frames,
    pred_frames,
    expected_ap,
):
    evaluator = WaymoEvaluator()
    class_names = (box["name"] for gt_frame in gt_frames for box in gt_frame["boxes"])
    config = WaymoEvaluationConfig(
        class_thresholds=[
            ClassThresholdConfig(
                class_name=name,
                iou_threshold=0.5,
            )
            for name in class_names
        ],
        num_score_thresholds=3,
        pr_curve_density=0.05,
        pr_rounding=6,
        min_gt_lidar_points=0,
    )

    darts = make_darts(
        scene_record=scene_record,
        sample_record=sample_record,
        sample_annotation_record=sample_annotation_record,
        instance_record=instance_record,
        category_record=category_record,
        gt_frames=gt_frames,
    )

    annotations = DARTSAnnotations(
        sequences={
            "scene_1": [
                Frame(
                    sample_token=frame["sample"],
                    boxes=[
                        make_box(
                            center=box["center"],
                            size=box.get("size", [1, 1, 1]),
                            name=box["name"],
                            score=box.get("score", 1),
                        )
                        for box in frame["boxes"]
                    ],
                )
                for frame in pred_frames
            ]
        }
    )

    results = evaluator.evaluate(
        darts=darts,
        annotations=annotations,
        config=config,
    )
    assert results.m_ap == expected_ap
