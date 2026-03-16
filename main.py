"""Main module for testing purposes."""

import logging
import darts.evaluation as de

from darts import DARTS, EvaluateRegistry
logger = logging.getLogger(__name__)


def main() -> None:
    """Main method for testing purposes."""
    #logging.basicConfig(level=logging.ERROR)
    logging.basicConfig(level=logging.INFO)
    darts = DARTS("/data/dataset", "dataset_exported")
    #darts.verify_integrity()
    #logger.info(darts.category.all())
    #logger.info(darts)
    #logger.info(EvaluateRegistry.available())
    de.register_eval_one()  # registers only EvalOne
    #logger.info(EvaluateRegistry.available())
    evaluator_cls = EvaluateRegistry.get("EvalOne")
    evaluator = evaluator_cls()
    #logger.info(evaluator.evaluate(darts))
    darts_filtered = darts.filter_scenes({
    "or": [
        {"intersection_y": 1},
        {"road_geometry": ["curve", "straight"]}
    ]
    })


if __name__ == "__main__":
    main()
