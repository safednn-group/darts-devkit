"""Data classes."""

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class PointCloud(ABC):
    """Abstract class for manipulating and viewing point clouds.

    Every point cloud (lidar and radar) consists of points where:

    - Dimensions 0, 1, 2 represent x, y, z. These are modified when the point cloud is rotated or translated.

    - All other dimensions are optional. Hence these have to be manually modified if the reference frame changes.

    Args:
        points: <np.float: d, n>. d-dimensional input point cloud matrix
    """

    def __init__(self, points: np.ndarray) -> None:
        """Initialize a point cloud and check it has the correct dimensions."""
        if points.shape[0] != self.nbr_dims():
            msg = f"Error: Pointcloud points must have format: {self.nbr_dims()} x n"
            logger.error(msg)
            raise ValueError(msg)
        self.points = points

    @staticmethod
    @abstractmethod
    def nbr_dims() -> int:
        """Returns the number of dimensions.

        Returns:
            number of dimensions
        """

    @classmethod
    @abstractmethod
    def from_file(cls, file_name: str) -> PointCloud:
        """Loads point cloud from disk.

        Args:
            file_name: Path of the pointcloud file on disk
        Returns:
            PointCloud instance
        """

    def nbr_points(self) -> int:
        """Returns the number of points.

        Returns:
            number of points
        """
        return self.points.shape[1]


class LidarPointCloud(PointCloud):
    """Class for manipulating and viewing lidar point clouds.

    Args:
        points: <np.float: 4, n>. 4-dimensional input point cloud matrix
    """

    @staticmethod
    def nbr_dims() -> int:
        """Returns the number of points.

        Returns:
            number of points
        """
        return 4

    @classmethod
    def from_file(cls, file_name: str) -> LidarPointCloud:
        """Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).

        Args:
            file_name: Path of the pointcloud file on disk
        Returns:
            LidarPointCloud instance (x, y, z, intensity)
        """
        if not file_name.endswith(".bin"):
            msg = f"Unsupported filetype {file_name}"
            logger.error(msg)
            raise TypeError(msg)

        scan = np.fromfile(file_name, dtype=np.float32)
        points = scan.reshape((-1, 5))[:, : cls.nbr_dims()]
        return cls(points.T)
