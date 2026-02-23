"""Module for Table class."""

from dataclasses import dataclass
from typing import Self, TypeVar

from tqdm import tqdm

T = TypeVar("T", bound="TableRecord")
"""`T` is a type variable representing any subclass of :class:`TableRecord`."""


@dataclass(slots=True)
class TableRecord:
    """Base class for records that can be loaded from a dictionary.

    Subclasses must implement the `from_dict` method.
    """

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Construct an instance from a dictionary.

        Args:
            data: A dictionary containing the record fields.

        Returns:
            An instance of the subclass.

        """
        return cls(**data)


class Table[T]:
    """In-memory indexed collection of dataset records.

    `T` is a type variable representing any subclass of :class:`TableRecord`.

    This class provides:
        - O(1) lookup by a specified key.
        - Iteration over all records.

    Note:
        This class is responsible only for storage and indexing.
        Relationship queries should be handled by higher-level components.

    Args:
        records: List of instances of type `T`.
        key: Attribute name to use as the dictionary key for O(1) lookup (default: "token").
        verbose: Whether to show a progress bar while indexing.
    """

    def __init__(self, records: list[T], key: str = "token", verbose: bool = False) -> None:
        """Create a table with instances of class `T` and prepare indexing by a key."""
        self._records = records
        iterable = tqdm(records, desc="Building index", unit="records") if verbose else records
        self._index = {getattr(r, key): r for r in iterable}

    def get(self, token: str) -> T:
        """Retrieve a record by its key.

        Args:
            token: The key value to look up.

        Returns:
            The record of type `T` corresponding to the key.

        Raises:
            KeyError: If no record exists with the given key.
        """
        return self._index[token]

    def all(self) -> list[T]:
        """Return all records in the table.

        Returns:
            A list of all instances of type `T`.
        """
        return self._records
