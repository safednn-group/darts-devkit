"""Methods to register classes that inherit from VisualizeInterface."""


def register_rerun_visualizer() -> None:
    """Register RerunVisualizer if installed."""
    try:
        from .rerun_visualizer import RerunVisualizer  # noqa: F401 PLC0415
    except ImportError as e:
        msg = "RerunVisualizer is not installed. Install with `uv pip install darts[rerun_visualizer]`"
        raise RuntimeError(msg) from e
