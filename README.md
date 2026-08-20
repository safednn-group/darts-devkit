# DARTS-devkit

A development toolkit for working with the **DARTS dataset**, providing utilities for dataset access, evaluation, and visualization of annotations.
To download run ```pip install darts-devkit```.
## Overview

- [Features](#features)
- [Local development](#local-development)

## Features

Below are presented simple examples for some features of the DARTS-devkit. For more examples download source code and build documentation.
### Dataset traversal
```
    from darts_devkit import DARTS

    darts=DARTS("dataset_root_path", "dataset_version")
    scenes = darts.scene.all()
    for scene in scenes:
        sample = darts.sample.get(scene.first_sample_token)
        for ann_token in sample.anns:
            print(darts.sample_annotation.get(ann_token))
```
### Dataset evaluation
```
uv pip install darts_devkit[polygon_overlap_evaluator]
```
```
    import json
    import darts_devkit.evaluation as ev
    from darts_devkit import DARTS, EvaluateRegistry
    from darts_devkit.dataset.evaluation_models import DARTSAnnotations
    from darts_devkit.evaluation.polygon_overlap_evaluator import PolygonOverlapEvaluationConfig, ClassThresholdConfig

    ev.register_polygon_overlap_evaluator()
    evaluator = EvaluateRegistry().get("PolygonOverlapEvaluator")()
    darts=DARTS("dataset_root_path", "dataset_version")
    config = PolygonOverlapEvaluationConfig(class_thresholds=[ClassThresholdConfig(class_name="multi_track_vehicle.car",  iou_threshold=0.5)], 
                                                             num_score_thresholds=10, pr_curve_density=0.05, pr_rounding=6, min_gt_lidar_points=0)

    with open('annotations.json', 'r') as file:
        annotations = DARTSAnnotations(**json.load(file))

    results = evaluator.evaluate(darts, annotations, config)
    print(results)
```

### Dataset visualization
```
uv pip install darts_devkit[rerun_visualizer]
```
```
    from darts_devkit import DARTS, VisualizeRegistry
    import darts_devkit.visualization as ve

    darts=DARTS("dataset_root_path", "dataset_version")

    ve.register_rerun_visualizer()
    visualizer_cls = VisualizeRegistry.get("RerunVisualizer")
    visualizer = visualizer_cls()
    visualizer.visualize(darts, "scene_token")
```
## Local development

### First installation

```
# One time install of 'uv'
curl -LsSf https://astral.sh/uv/install.sh | sh

cd darts-devkit
uv sync --all-extras
```

### Run examples

```
# Editable for development - changes in package reflected instantly
uv run main.py
# Non-Editable - can delete src/darts_devkit but code will still work
uv pip install .[dev,docs]
source .venv/bin/activate
python main.py
```

### Quality check

```
make format
make check
```

### Unit tests check
```
make test
```

### Building documentation

```
make docs
```