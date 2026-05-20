"""Main module for testing purposes."""

import logging
import darts.evaluation as ev
import json

from darts import DARTS, EvaluateRegistry
from darts.dataset.user_models import DARTSAnnotations
from darts.evaluation.waymo_evaluator import WaymoEvaluationConfig, ClassThresholdConfig
logger = logging.getLogger(__name__)


def main() -> None:
    """Main method for testing purposes."""
    logging.basicConfig(level=logging.INFO)
    darts = DARTS("/data/dataset", "dataset_exported")

    ev.register_waymo_evaluator()
    with open('/data/dataset/annotations.json', 'r') as file:
        data = json.load(file)
    annotations = DARTSAnnotations(**data)
    evaluator_cls = EvaluateRegistry.get("WaymoEvaluator")
    evaluator = evaluator_cls()
    config = WaymoEvaluationConfig(class_thresholds=[ClassThresholdConfig(class_name="multi_track_vehicle.car",  iou_threshold=0.7)], num_score_thresholds=10, pr_curve_density=0.05, pr_rounding=6, min_gt_lidar_points=0)
    results = evaluator.evaluate(darts, annotations, config)
    print('results', results)
if __name__ == "__main__":
    main()
