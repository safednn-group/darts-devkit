"""Main module for testing purposes."""

import logging
import darts.evaluation as de
import darts.visualization as ve

from darts import DARTS, EvaluateRegistry, VisualizeRegistry
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
    ve.register_rerun_visualizer()
    visualizer_cls = VisualizeRegistry.get("RerunVisualizer")
    visualizer = visualizer_cls()
    visualizer.visualize(darts, "36d2d4317d1847bd87ee94f305bcee8f")

if __name__ == "__main__":
    main()
