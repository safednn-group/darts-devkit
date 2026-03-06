"""DEMO with EvalTwo class."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .registry import EvaluateInterface, register_evaluator

if TYPE_CHECKING:
    from darts.dataset.darts import DARTS


@register_evaluator("EvalTwo")
class EvalTwo(EvaluateInterface):
    """EvalTwo class."""

    def evaluate(self, darts: DARTS) -> str:
        """Evaluate annotations with method EvalTwo.

        :param darts: DARTS database
        :type darts: DARTS
        :return: evaluation results
        :rtype: str
        """
        return f"EvalTwo:{darts}"
