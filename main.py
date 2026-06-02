"""Main module for testing purposes."""

import logging
from darts_devkit import DARTS
logger = logging.getLogger(__name__)


def main() -> None:
    """Main method for testing purposes."""
    logging.basicConfig(level=logging.INFO)
    darts=DARTS("/data/dataset", "dataset_exported")
    scenes = darts.scene.all()
    for scene in scenes:
        sample = darts.sample.get(scene.first_sample_token)
        for ann_token in sample.anns:
            print(darts.sample_annotation.get(ann_token))
if __name__ == "__main__":
    main()
