"""Main module for testing purposes."""

import logging
import darts.evaluate as de

from darts import DARTS
from darts.core.evaluate import EvaluateRegistry
logger = logging.getLogger(__name__)


def main() -> None:
    """Main method for testing purposes."""
    darts = DARTS("/data/dataset", "dataset_exported", True, True)
    #logger.info(darts.calibrated_sensor.all())
    logger.info(darts)
    logger.info(EvaluateRegistry.available())
    de.register_eval_one()  # registers only EvalOne
    logger.info(EvaluateRegistry.available())
    evaluator_cls = EvaluateRegistry.get("EvalOne")
    evaluator = evaluator_cls()
    logger.info(evaluator.evaluate(darts))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
