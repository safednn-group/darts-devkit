"""Main module for testing purposes."""

import logging
import darts_devkit.evaluation as ev
import json
import cProfile
import pstats
from darts_devkit import DARTS, EvaluateRegistry
from darts_devkit.evaluation.evaluation_models import DARTSAnnotations
from darts_devkit.evaluation.polygon_overlap_evaluator import PolygonOverlaEvaluationConfig, ClassThresholdConfig
logger = logging.getLogger(__name__)


def main() -> None:
    """Main method for testing purposes."""
    logging.basicConfig(level=logging.INFO)
    darts = DARTS("/data/dataset", "dataset_copy")

    ev.register_polygon_overlap_evaluator()
    with open('test_annotations.json', 'r') as file:
        data = json.load(file)
    annotations = DARTSAnnotations(**data)
    evaluator_cls = EvaluateRegistry.get("PolygonOverlapEvaluator")
    evaluator = evaluator_cls()
    config = PolygonOverlaEvaluationConfig(class_thresholds=[ClassThresholdConfig(class_name="multi_track_vehicle.car",  iou_threshold=0.0),
                                                             ClassThresholdConfig(class_name="multi_track_vehicle.truck",  iou_threshold=0.0)], 
                                                             num_score_thresholds=10, pr_curve_density=0.05, pr_rounding=6, min_gt_lidar_points=0)
    results = evaluator.evaluate(darts, annotations, config)
    print('results', results)
if __name__ == "__main__":
    cProfile.run("main()", "profile.out")

    p = pstats.Stats("profile.out")
    p.sort_stats("cumtime").print_stats(30)
