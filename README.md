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
    darts=DARTS("dataset_root_path", "dataset_version")
    scenes = darts.scene.all()
    for scene in scenes:
        sample = darts.sample.get(scene.first_sample_token)
        for ann_token in sample.anns:
            print(darts.sample_annotation.get(ann_token))
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