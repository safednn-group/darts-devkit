"""Main module for testing purposes."""

import logging
import darts.evaluation as ev
import json

from darts import DARTS, EvaluateRegistry
from darts.dataset.user_models import DARTSAnnotations
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
    evaluator.evaluate(darts, annotations)
if __name__ == "__main__":
    main()
