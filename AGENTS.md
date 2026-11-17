# AGENTS.md — DARTS-devkit

## Purpose and working rules

DARTS-devkit is a public Python toolkit for loading and traversing DARTS scenes, evaluating predictions,
and visualizing sensor data and annotations. Documentation and notebooks demonstrate these workflows.
The distribution is `darts-devkit`; the import package is `darts_devkit`.

- The team limits LLM use to demanding, explicitly assigned tasks. Treat generated changes as proposals
  requiring verification and human review. Do not run additional models or subagents without an explicit instruction.
- Stay within the assigned scope. Preserve existing edits and untracked files; avoid unrelated refactoring,
  dependency updates, and cleanup. Do not commit, push, or publish packages without an instruction.
- Start with `git status --short`, then read the relevant README, docstrings, callers, models, and tests.
  Check `pyproject.toml`, `Makefile`, and CI for current commands. Report discrepancies rather than guessing.
- Inspect source and notebook cells as text before executing them. Read only the data needed for the task;
  avoid loading entire datasets, binary scans, or large notebook outputs into context.

## Code map and architecture

| Location | Responsibility |
| --- | --- |
| `src/darts_devkit/__init__.py` | Public exports: `DARTS`, `EvaluateRegistry`, `VisualizeRegistry`, `__version__`. |
| `src/darts_devkit/dataset/` | Dataset access (`darts.py`), dataclass records (`dataset_models.py`), indexed collections (`record_collection.py`), Pydantic evaluation models (`evaluation_models.py`), and LiDAR loading (`pointcloud.py`). |
| `src/darts_devkit/evaluation/` | Evaluator interface/registry and polygon-overlap evaluation: volumetric IoU, matching, PR curves, AP, and mAP. |
| `src/darts_devkit/visualization/` | Visualizer interface/registry and the optional Rerun implementation. |
| `tests/` | Synthetic fixtures in `conftest.py`; dataset, evaluation, and visualization tests under `tests/darts/`. |
| `docs/source/`, `notebooks/`, `images/` | Sphinx documentation, usage examples, and supporting images. |
| `pyproject.toml`, `Makefile`, `.github/` | Packaging, dependencies, quality checks, and GitHub Actions CI. |

Keep dataset access, evaluation, and visualization separate. Extend the existing interfaces and registries
rather than placing backend-specific logic in `DARTS`. Reuse local models and helpers before adding new ones.

## Environment and optional extensions

- Use `uv` and the project `.venv`. From the repository root, `uv sync --all-extras` prepares the development
  environment used by CI. Do not install dependencies into the system Python.
- Python support is declared in `pyproject.toml`; CI currently selects Python 3.9. Preserve compatibility
  when changing syntax or dependencies. Do not update dependency bounds or package versions incidentally.
- Optional extras are `polygon_overlap_evaluator`, `rerun_visualizer`, and `docs`; development tools are in
  the `dev` dependency group. Basic dataset access must work without evaluator or viewer dependencies.
- Preserve explicit registration through `register_polygon_overlap_evaluator()` and `register_rerun_visualizer()`.
  Registry `get()` returns a class for the caller to instantiate. Keep registration names and missing-extra errors useful.
- The build backend is `uv_build`. Keep packaging configuration in `pyproject.toml`; `uv.lock` is currently ignored.
  Both synchronization and `uv run` can install dependencies or access the network.

## Public API and data contracts

- Write code and documentation in English. Preserve public imports, signatures, return types, exceptions,
  registry names, and serialized formats. Describe compatibility impact when a task requires changing them.
- Dataset records are dataclasses; prediction payloads use Pydantic `Box`, `Frame`, and `DARTSAnnotations`.
  Preserve their respective validation and serialization rules, including rejection of extra fields in `Box`.
- Keep supplied token strings and record relationships intact. `RecordCollection` indexes by `token` by default;
  missing lookups raise `KeyError`. Preserve empty-string markers at the ends of sample chains.
- JSON tables are loaded from `root / version`; sensor filenames are relative to `root`. Respect derived fields
  marked with `metadata={"dump": False}`. Filtered datasets can share records with their source; do not assume deep copies.
- Metadata queries support `and`, `or`, and `not`; lists represent a disjunction. Splits contain scene names,
  while prediction sequences use scene tokens. Preserve filtering semantics and held-out split boundaries.
- `save(version_name)` writes JSON tables to a new version directory and rejects an existing destination.
  It does not copy sensor binaries or provide a transactional multi-file export. Never overwrite source datasets.

## Geometry, metrics, and visualization

- Check units, coordinate frames, axis directions, dimension order, and transformation direction.
  Dataset quaternions use `[w, x, y, z]`; Rerun uses `xyzw`. Timestamps are Unix microseconds and convert to UTC.
- LiDAR binary files contain five float32 values per point `(x, y, z, intensity, ring index)`; the loader retains
  the first four as a `4 x N` matrix. Preserve shapes, field interpretation, and modality checks.
- Predictions must preserve scene/sample tokens and frame ordering. Do not silently drop, reorder, or fabricate
  frames to conceal alignment errors. Test missing predictions and empty frames explicitly when changing this behavior.
- Preserve the evaluator's volumetric IoU, Hungarian matching, configured classes, threshold comparisons,
  PR processing, rounding, and AP/mAP aggregation. Explain numerical changes and verify known TP/FP/FN/AP cases.
- Dataset-based evaluation can filter ground truth by `min_gt_lidar_points`; evaluation from supplied annotation
  files does not have those per-box counts. Do not assume both paths apply identical filtering.
- `RerunVisualizer.visualize()` starts a viewer through `rr.spawn()`. Mock viewer calls in unit tests;
  launch the actual viewer only when needed for the task. Keep startup out of imports and dataset loading.

## Errors, logging, and data safety

- Use `pathlib`, explicit UTF-8 for new text I/O, and appropriate resource cleanup. Validate formats, shapes,
  references, and paths at input boundaries. Use exceptions for invalid user input rather than `assert`.
- Catch specific exceptions where recovery is possible; otherwise propagate useful table/file/token context
  and preserve causes. Never turn invalid data or missing dependencies into empty successful results.
- Use module-level `logging.getLogger(__name__)` and parameterized messages. Let the calling application
  configure logging; do not add root handlers, unsolicited output, or per-point log floods.
- Keep reads and computations from modifying sensor files. Restrict writes to intended output locations,
  check path containment where required, and clean up only task-owned temporary files.
- `verify_integrity()` compares sensor-file MD5 checksums; it does not authenticate a dataset or validate every
  schema relationship. Do not bypass mismatches or rewrite checksums to hide corruption.
- Do not read or log secrets, use API keys, or upload private datasets and identifiable sensor data to external
  services. If access to a key becomes known, inform the user before executing further commands; do not search for keys.
- Public examples must work without private credentials or team-specific paths. Preserve licenses and attribution;
  the toolkit's MIT license does not establish redistribution rights for datasets or images.

## Formatting and validation

Run commands from the repository root after preparing the environment. Inspect recipes and fixtures first.

| Command | Behavior |
| --- | --- |
| `make format` | Ruff formats `src/` and `tests/`, then applies lint fixes to `src/`. Modifies files. |
| `make check` | Ruff lint and mypy on `src/`; does not format files or run tests. |
| `make test` | Pytest with randomized order, package coverage, and an HTML coverage report. |
| `make docs` | Deletes `docs/build` and builds Sphinx HTML into `docs/build/html/`. |

Follow Ruff configuration in `pyproject.toml`: 120-character lines, Google docstring conventions, and the
configured lint rules. Justify narrow suppressions rather than disabling checks globally.

Use synthetic records, small JSON fixtures, `tmp_path`, numerical tolerances, and mocked Rerun calls.
Tests must not depend on real datasets, network access, GUI sessions, or execution order. Reset registry state
when a test changes it. Loader/serialization changes should exercise real small files as well as mocked objects.
Check base-package imports without extras when changing optional dependency boundaries.

CI runs checks, tests, and documentation builds; it does not execute notebooks. Report exact validation results
and distinguish failed, skipped, and unexecuted checks. Do not claim a real viewer or dataset was tested from mocks.

## Documentation, notebooks, and handoff

- Update README examples, API docstrings, Sphinx pages, and notebooks alongside relevant behavior changes.
  Document parameters, results, exceptions, units, required extras, and effects such as file writes or viewer startup.
- Keep notebooks executable from top to bottom with explicit dataset roots/versions and no hidden kernel state.
  Inspect installation/download/write cells before running them. Avoid unrelated metadata/output churn and
  never commit sensitive data, personal paths, or large generated outputs.
- Check examples against implemented APIs. Review Sphinx warnings; `make docs` does not currently use `-W`.
  Do not edit generated documentation or add datasets, caches, coverage reports, `.venv`, or build artifacts to changes.
- For release-related work, check version consistency across `pyproject.toml`, `src/darts_devkit/version.py`, and
  `docs/source/conf.py`. Report disagreements instead of guessing the intended release version.
- Before delivery, review the diff, `git diff --check`, and status. Summarize the change, validation performed,
  remaining limitations, and any public API or metric impact. Keep this file aligned with the project's actual tools.
- Every file, class or function/method should have a docstring in a format similar to other files.
