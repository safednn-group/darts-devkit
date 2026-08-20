"""Module containing data types used by evaluation."""

from __future__ import annotations

from typing import Annotated

from annotated_types import Len
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat


class Box(BaseModel):
    """Data type representing a bounding box in DARTS format."""

    center: Annotated[list[float], Len(min_length=3, max_length=3)]
    """Center of a box [x, y, z]."""
    size: Annotated[list[PositiveFloat], Len(min_length=3, max_length=3)]
    """Size of a box [x, y, z]."""
    orientation: Annotated[list[Annotated[float, Field(ge=-1, le=1)]], Len(min_length=4, max_length=4)]
    """Quaternion rotation [w, x, y, z]."""
    name: str
    """Class name."""
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    """Detector confidence score."""
    model_config = ConfigDict(extra="forbid")


class Frame(BaseModel):
    """Data type representing a frame in DARTS format."""

    sample_token: str
    boxes: list[Box]


class DARTSAnnotations(BaseModel):
    """Data type representing annotations in DARTS format."""

    sequences: dict[str, list[Frame]]
