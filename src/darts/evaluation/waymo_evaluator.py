"""DARTS evaluation based on The Waymo Open Dataset."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

import numpy as np
from pydantic import BaseModel, Field
from pyquaternion import Quaternion
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from darts.evaluation.registry import ClassResults, Results

from .registry import EvaluateInterface, register_evaluator

if TYPE_CHECKING:
    from darts.dataset.darts import DARTS
    from darts.dataset.dataset_models import SampleAnnotation
    from darts.dataset.user_models import Box, DARTSAnnotations

K_EPSILON = 1e-10
K_MIN_BOX_DIM = 1e-2
logger = logging.getLogger(__name__)


class ClassThresholdConfig(BaseModel):
    """IoU threshold for a single class."""

    class_name: str
    iou_threshold: Annotated[float, Field(ge=0.0, le=1.0)]


class WaymoEvaluationConfig(BaseModel):
    """User configuration for Waymo-style evaluation."""

    class_thresholds: list[ClassThresholdConfig]
    num_score_thresholds: Annotated[int, Field(ge=1)]
    pr_curve_density: Annotated[float, Field(ge=0.0, le=1.0)]
    pr_rounding: Annotated[int, Field(ge=0)]
    min_gt_lidar_points: Annotated[int, Field(ge=0)]


@dataclass
class WaymoBox:
    """WaymoBox."""

    name: str
    polygon: Polygon
    z_min: float
    z_max: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    size: list[float]
    num_lidar_pts: int
    score: float


@register_evaluator("WaymoEvaluator")
class WaymoEvaluator(EvaluateInterface[WaymoEvaluationConfig]):
    """WaymoEvaluator class."""

    def evaluate(self, darts: DARTS, annotations: DARTSAnnotations, config: WaymoEvaluationConfig) -> Results:
        """Evaluate annotations with WaymoEvaluator.

        :param darts: DARTS database
        :param annotations: Annotations created by user
        :return: evaluation results
        """
        gts_by_frame, dts_by_frame = self._get_boxes_by_frame(darts, annotations)
        class_results: list[ClassResults] = []
        m_ap = 0.0
        for class_cfg in config.class_thresholds:
            gts_class_by_frame = [
                [gt for gt in gts if gt.name == class_cfg.class_name and gt.num_lidar_pts >= config.min_gt_lidar_points]
                for gts in gts_by_frame
            ]
            dts_class_by_frame = [[dt for dt in dts if dt.name == class_cfg.class_name] for dts in dts_by_frame]
            class_result = self._get_class_result(gts_class_by_frame, dts_class_by_frame, config, class_cfg)
            class_results.append(class_result)
            m_ap += class_result.ap
        m_ap /= len(class_results)
        return Results(class_results=class_results, m_ap=m_ap)

    def _get_class_result(
        self,
        gts_class_by_frame: list[list[WaymoBox]],
        dts_class_by_frame: list[list[WaymoBox]],
        config: WaymoEvaluationConfig,
        class_cfg: ClassThresholdConfig,
    ) -> ClassResults:
        pr_curve = []
        thresholds = np.linspace(0.0, 1.0, config.num_score_thresholds)
        tp_list = []
        fp_list = []
        fn_list = []
        for score_threshold in thresholds:
            tp = 0
            fp = 0
            fn = 0

            for frame_id in range(len(gts_class_by_frame)):
                gt_class = gts_class_by_frame[frame_id]
                dt_class = [dt for dt in dts_class_by_frame[frame_id] if dt.score >= score_threshold]

                if len(dt_class) == 0:
                    fn += len(gt_class)
                    continue

                if len(gt_class) == 0:
                    fp += len(dt_class)
                    continue

                # IoU matrix per frame only
                iou_matrix = np.zeros((len(dt_class), len(gt_class)), dtype=np.float32)

                for i, dt in enumerate(dt_class):
                    for j, gt in enumerate(gt_class):
                        iou = self._compute_iou(dt, gt)
                        if iou >= class_cfg.iou_threshold:
                            iou_matrix[i, j] = iou
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)

                matched_gt = set()
                matched_dt = set()

                for r, c in zip(row_ind, col_ind, strict=True):
                    if iou_matrix[r, c] >= class_cfg.iou_threshold:
                        tp += 1
                        matched_gt.add(c)
                        matched_dt.add(r)

                fp += len(dt_class) - len(matched_dt)
                fn += len(gt_class) - len(matched_gt)

            precision = tp / (tp + fp + K_EPSILON)
            recall = tp / (tp + fn + K_EPSILON)
            pr_curve.append((precision, recall))
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
        pr_curve = self._sort_and_fill_pr_curve_gaps(pr_curve, config.pr_curve_density, config.pr_rounding)
        ap = self._calculate_ap(pr_curve, config.pr_rounding)
        return ClassResults(
            class_name=class_cfg.class_name, fp_list=fp_list, fn_list=fn_list, tp_list=tp_list, ap=ap, pr_curve=pr_curve
        )

    def _calculate_ap(self, pr_curve: list[tuple[float, float]], pr_rounding: int) -> float:
        ap = 0.0
        for i in range(1, len(pr_curve)):
            ap += 0.5 * (pr_curve[i - 1][1] - pr_curve[i][1]) * (pr_curve[i - 1][0] + pr_curve[i][0])
        return round(ap, pr_rounding)

    def _sort_and_fill_pr_curve_gaps(
        self,
        pr_curve: list[tuple[float, float]],
        pr_curve_density: float,
        pr_rounding: int,
    ) -> list[tuple[float, float]]:

        if not pr_curve:
            return []

        # build recall -> precision map
        recall_precision = {0.0: 1.0}

        for p_, r_ in pr_curve:
            r = round(r_, pr_rounding)
            p = round(p_, pr_rounding)
            recall_precision[r] = max(recall_precision.get(r, 0.0), p)

        # sort by recall DESC
        items = sorted(recall_precision.items(), key=lambda x: x[0], reverse=True)

        # reverse traversal + gap filling (core logic)
        precision_recall: list[tuple[float, float]] = []

        last_recall = items[0][0]
        max_precision = 0.0

        for r, p in items:
            # Fill recall gaps (Waymo max_recall_delta logic)
            while last_recall - r > pr_curve_density + K_EPSILON:
                last_recall -= pr_curve_density
                precision_recall.append((max_precision, round(last_recall, pr_rounding)))

            max_precision = max(max_precision, p)
            precision_recall.append((max_precision, r))
            last_recall = r

        # 4. Final correction step
        atleast_two = 2
        if len(precision_recall) >= atleast_two:
            prev_p = precision_recall[-2][0]
            last_r = precision_recall[-1][1]
            precision_recall[-1] = (prev_p, last_r)

        return precision_recall

    def _get_boxes_by_frame(
        self, darts: DARTS, annotations: DARTSAnnotations
    ) -> tuple[list[list[WaymoBox]], list[list[WaymoBox]]]:
        gts_by_frame: list[list[WaymoBox]] = []
        dts_by_frame: list[list[WaymoBox]] = []
        for scene_token in annotations.sequences:
            for sample_idx, sample in enumerate(darts.get_samples_from_scene(scene_token)):
                gts_by_frame.append(
                    [
                        self._sample_annotation_to_waymo_box(darts, gt)
                        for gt in darts.get_annotations_from_samples([sample])
                    ]
                )
                if annotations.sequences[scene_token][sample_idx].sample_token != sample.token:
                    msg = "frames are in bad order compared to dataset"
                    logger.error(msg)
                    raise IndexError(msg)
                dts_by_frame.append(
                    [self._box_to_waymo_box(dt) for dt in annotations.sequences[scene_token][sample_idx].boxes]
                )
        return gts_by_frame, dts_by_frame

    def _box_to_waymo_box(self, box: Box) -> WaymoBox:
        polygon, x_min, x_max, y_min, y_max = self._to_polygon(
            translation=box.center, size=box.size, rotation=box.orientation
        )
        return WaymoBox(
            name=box.name,
            polygon=polygon,
            size=box.size,
            z_min=box.center[2] - box.size[2] / 2,
            z_max=box.center[2] + box.size[2] / 2,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            score=box.score,
            num_lidar_pts=-1,
        )

    def _sample_annotation_to_waymo_box(self, darts: DARTS, sample_annotation: SampleAnnotation) -> WaymoBox:
        polygon, x_min, x_max, y_min, y_max = self._to_polygon(
            translation=list(sample_annotation.translation),
            size=list(sample_annotation.size),
            rotation=list(sample_annotation.rotation),
        )
        return WaymoBox(
            name=darts.get_category_from_annotation(sample_annotation.token).name,
            polygon=polygon,
            size=list(sample_annotation.size),
            z_min=sample_annotation.translation[2] - sample_annotation.size[2] / 2,
            z_max=sample_annotation.translation[2] + sample_annotation.size[2] / 2,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            score=sample_annotation.score,
            num_lidar_pts=sample_annotation.num_lidar_pts,
        )

    def _to_polygon(
        self, translation: list[float], size: list[float], rotation: list[float]
    ) -> tuple[Polygon, float, float, float, float]:
        cx, cy, _ = translation
        length_x, length_y, _ = size

        q = Quaternion(
            w=rotation[0],
            x=rotation[1],
            y=rotation[2],
            z=rotation[3],
        )

        yaw = q.yaw_pitch_roll[0]

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        local = [
            (length_x / 2, -length_y / 2),
            (length_x / 2, length_y / 2),
            (-length_x / 2, length_y / 2),
            (-length_x / 2, -length_y / 2),
        ]
        pts = []
        for x, y in local:
            rx = cos_y * x - sin_y * y + cx
            ry = sin_y * x + cos_y * y + cy
            pts.append((rx, ry))

        x_min = min(x for x, _ in pts)
        x_max = max(x for x, _ in pts)
        y_min = min(y for _, y in pts)
        y_max = max(y for _, y in pts)
        return Polygon(pts), x_min, x_max, y_min, y_max

    def _probable_overlap(self, b1: WaymoBox, b2: WaymoBox) -> bool:

        overlap_min_z = max(b1.z_min, b2.z_min)
        overlap_max_z = min(b1.z_max, b2.z_max)

        overlap_exists_z = overlap_min_z <= overlap_max_z

        overlap_min_x = max(b1.x_min, b2.x_min)
        overlap_max_x = min(b1.x_max, b2.x_max)

        overlap_exists_x = overlap_min_x <= overlap_max_x

        overlap_min_y = max(b1.y_min, b2.y_min)
        overlap_max_y = min(b1.y_max, b2.y_max)

        overlap_exists_y = overlap_min_y <= overlap_max_y
        return overlap_exists_x and overlap_exists_y and overlap_exists_z

    def _compute_iou(self, b1: WaymoBox, b2: WaymoBox) -> float:
        for size in b1.size:
            if size < K_MIN_BOX_DIM:
                return 0.0
        for size in b2.size:
            if size < K_MIN_BOX_DIM:
                return 0.0

        is_probable_overlap = self._probable_overlap(b1, b2)
        if not is_probable_overlap:
            return 0.0

        intersection_area = b1.polygon.intersection(b2.polygon).area
        intersection_volume = intersection_area * (min(b1.z_max, b2.z_max) - max(b1.z_min, b2.z_min))
        b1_volume = b1.size[0] * b1.size[1] * b1.size[2]
        b2_volume = b2.size[0] * b2.size[1] * b2.size[2]
        union_volume = b1_volume + b2_volume - intersection_volume
        if union_volume < K_EPSILON:
            return 0.0
        iou = intersection_volume / union_volume
        return max(min(iou, 1), 0)
