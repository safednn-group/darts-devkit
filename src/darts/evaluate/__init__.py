"""Methods to register classes that inherit from EvaluateInterface."""


def register_eval_one() -> None:
    """Register EvalOne if installed."""
    try:
        from .eval_one import EvalOne  # noqa: F401 PLC0415
    except ImportError as e:
        msg = "EvalOne is not installed. Install with `uv pip install darts[eval_one]`"
        raise RuntimeError(msg) from e


def register_eval_two() -> None:
    """Register EvalTwo if installed."""
    try:
        from .eval_two import EvalTwo  # noqa: F401 PLC0415
    except ImportError as e:
        msg = "EvalTwo is not installed. Install with `uv pip install darts[eval_two]`"
        raise RuntimeError(msg) from e
