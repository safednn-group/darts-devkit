"""Methods to register classes that inherit from EvaluateInterface."""


def register_waymo_evaluator() -> None:
    """Register WaymoEvaluator if installed."""
    try:
        from .waymo_evaluator import WaymoEvaluator  # noqa: F401 PLC0415
    except ImportError as e:
        msg = "WaymoEvaluator is not installed. Install with `uv pip install darts[waymo_evaluator]`"
        raise RuntimeError(msg) from e
