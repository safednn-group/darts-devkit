"""Methods to register classes that inherit from EvaluateInterface."""


def register_polygon_overlap_evaluator() -> None:
    """Register PolygonOverlapEvaluator if installed."""
    try:
        from .polygon_overlap_evaluator import PolygonOverlapEvaluator  # noqa: F401 PLC0415
    except ImportError as e:
        msg = "PolygonOverlapEvaluator is not installed. Install with `uv pip install darts[polygon_overlap_evaluator]`"
        raise RuntimeError(msg) from e
