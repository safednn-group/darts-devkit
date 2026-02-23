"""Package for evaluation registry and interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from darts.core.darts import DARTS


class EvaluateInterface(ABC):
    """Abstract class/Interface for evaluate method."""

    @abstractmethod
    def evaluate(self, darts: DARTS) -> str:
        """Abstract method for EvaluateInterface interface.

        Args:
            darts: DARTS database
        Returns:
            evaluation results
        """


class EvaluateRegistry:
    """Registry for classes that inherit from EvaluateInterface.

    This class manages registration and retrieval of evaluator classes
    by name.
    """

    _registry: ClassVar[dict[str, type[EvaluateInterface]]] = {}

    @classmethod
    def register(cls, name: str, impl: type[EvaluateInterface]) -> None:
        """Register a class that inherits from EvaluateInterface.

        Args:
            name: Name of the class to register.
            impl: Class that inherits from EvaluateInterface.

        Raises:
            ValueError: If a class is already registered with this name.
        """
        if name in cls._registry:
            msg = f"Evaluator '{name}' already registered"
            raise ValueError(msg) from None
        cls._registry[name] = impl

    @classmethod
    def get(cls, name: str) -> type[EvaluateInterface]:
        """Retrieve a registered class by its name.

        Args:
            name: Name of the registered class.

        Returns:
            The class that inherits from EvaluateInterface.

        Raises:
            KeyError: If no class is registered under the given name.
        """
        try:
            return cls._registry[name]
        except KeyError:
            msg = f"Unknown evaluator '{name}'. Available: {list(cls._registry)}"
            raise KeyError(msg) from None

    @classmethod
    def available(cls) -> list[str]:
        """List all registered class names.

        Returns:
            A list of names of available registered evaluator classes.
        """
        return list(cls._registry)


def register_evaluator(name: str) -> Callable[[type[EvaluateInterface]], type[EvaluateInterface]]:
    """Decorator to register a class as an evaluator.

    This decorator registers a class that inherits from EvaluateInterface
    in the EvaluateRegistry under the given name.

    Args:
        name: Name to register the evaluator under.

    Returns:
        A decorator that registers the class and returns it unchanged.
    """

    def decorator(cls: type[EvaluateInterface]) -> type[EvaluateInterface]:
        EvaluateRegistry.register(name, cls)
        return cls

    return decorator
