"""Package for visualization registry and interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from darts.dataset.darts import DARTS


class VisualizeInterface(ABC):
    """Abstract class/Interface for visualize method."""

    @abstractmethod
    def visualize(self, darts: DARTS, scene_token: str) -> None:
        """Abstract method for VisualizeInterface interface.

        Args:
            darts: DARTS database
            scene_token: Scene to visualize
        """


class VisualizeRegistry:
    """Registry for classes that inherit from VisualizeInterface.

    This class manages registration and retrieval of Visualizer classes
    by name.
    """

    _registry: ClassVar[dict[str, type[VisualizeInterface]]] = {}

    @classmethod
    def register(cls, name: str, impl: type[VisualizeInterface]) -> None:
        """Register a class that inherits from VisualizeInterface.

        Args:
            name: Name of the class to register.
            impl: Class that inherits from VisualizeInterface.

        Raises:
            ValueError: If a class is already registered with this name.
        """
        if name in cls._registry:
            msg = f"Visualizer '{name}' already registered"
            raise ValueError(msg) from None
        cls._registry[name] = impl

    @classmethod
    def get(cls, name: str) -> type[VisualizeInterface]:
        """Retrieve a registered class by its name.

        Args:
            name: Name of the registered class.

        Returns:
            The class that inherits from VisualizeInterface.

        Raises:
            KeyError: If no class is registered under the given name.
        """
        try:
            return cls._registry[name]
        except KeyError:
            msg = f"Unknown visualizer '{name}'. Available: {list(cls._registry)}"
            raise KeyError(msg) from None

    @classmethod
    def available(cls) -> list[str]:
        """List all registered class names.

        Returns:
            A list of names of available registered visualizer classes.
        """
        return list(cls._registry)


def register_visualizer(name: str) -> Callable[[type[VisualizeInterface]], type[VisualizeInterface]]:
    """Decorator to register a class as an visualizer.

    This decorator registers a class that inherits from VisualizeInterface
    in the VisualizeRegistry under the given name.

    Args:
        name: Name to register the visualizer under.

    Returns:
        A decorator that registers the class and returns it unchanged.
    """

    def decorator(cls: type[VisualizeInterface]) -> type[VisualizeInterface]:
        VisualizeRegistry.register(name, cls)
        return cls

    return decorator
