"""DARTS visualizations based on rerun."""

from __future__ import annotations

import io
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .registry import VisualizeInterface, register_visualizer

if TYPE_CHECKING:
    import PIL.ImageFile

    from darts_devkit.dataset.darts import DARTS
    from darts_devkit.dataset.dataset_models import Sample, SampleAnnotation, SampleAnnotation2D, SampleData
    from darts_devkit.dataset.pointcloud import LidarPointCloud
import matplotlib as mpl
import rerun as rr


@register_visualizer("RerunVisualizer")
class RerunVisualizer(VisualizeInterface):
    """RerunVisualizer class."""

    def visualize(self, darts: DARTS, scene_token: str) -> None:
        """Visualize dataset with the help of rerun app.

        :param darts: DARTS database
        :param scene_token: Scene to visualize
        """
        rr.init(scene_token)
        rr.spawn()
        scene = darts.scene.get(scene_token)
        sample_token = scene.first_sample_token
        sample_number = 1
        while sample_token != "":
            rr.set_time("frame", sequence=sample_number)
            sample = darts.sample.get(sample_token)
            sample_number += 1
            self._log_frame(darts, sample)
            sample_token = sample.next

    def _log_frame(self, darts: DARTS, sample: Sample) -> None:
        annotations = darts.get_annotations_from_samples([sample])
        self._log_annotations(darts, annotations)
        lidar_sample_datas = self._get_keyframe_sample_data(darts, sample, "lidar")
        lidar_pointclouds = self._get_lidar_pointclouds(darts, lidar_sample_datas)
        self._log_lidar_pointclouds(darts, lidar_sample_datas, lidar_pointclouds)

        camera_sample_datas = self._get_keyframe_sample_data(darts, sample, "camera")
        self._log_camera_images(darts, camera_sample_datas)
        annotations_2d = darts.get_annotations_2d_from_sample_datas(camera_sample_datas)
        self._log_annotations_2d(darts, annotations_2d)

    def _log_annotations_2d(self, darts: DARTS, annotations_2d: list[SampleAnnotation2D]) -> None:
        annotations_corners_per_camera: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
        annotations_labels_per_camera: dict[str, list[str]] = defaultdict(list)
        for annotation_2d in annotations_2d:
            sensor = darts.get_sensor_from_annotation_2d(annotation_2d.token)
            category = darts.get_category_from_annotation_2d(annotation_2d.token)
            annotations_corners_per_camera[sensor.channel].append(annotation_2d.corners)
            annotations_labels_per_camera[sensor.channel].append(category.name)
        for camera_channel, annotations_corners in annotations_corners_per_camera.items():
            rr.log(
                f"world/ego_vehicle/{camera_channel}/annotations",
                rr.Boxes2D(
                    array=np.array(annotations_corners),
                    array_format=rr.Box2DFormat.XYXY,
                    labels=annotations_labels_per_camera[camera_channel],
                ),
            )

    def _log_camera_images(self, darts: DARTS, camera_sample_datas: list[SampleData]) -> None:
        for camera_sample_data in camera_sample_datas:
            channel = camera_sample_data.channel
            img = darts.get_image(camera_sample_data)
            self._log_sensor_calibration(darts, camera_sample_data)
            self._log_camera(img, channel)

    def _log_camera(self, img: PIL.ImageFile.ImageFile, channel: str) -> None:
        image_bytes = io.BytesIO()
        img.save(image_bytes, format=img.format)
        rr.log(f"world/ego_vehicle/{channel}", rr.Image(np.asarray(img)))

    def _log_lidar_pointclouds(
        self, darts: DARTS, lidar_sample_datas: list[SampleData], lidar_pointclouds: dict[str, LidarPointCloud]
    ) -> None:
        for lidar_sample_data in lidar_sample_datas:
            self._log_sensor_calibration(darts, lidar_sample_data)
            self._log_ego_pose(darts, lidar_sample_data)
            self._log_lidar(
                lidar_pointclouds[lidar_sample_data.token].points.T, lidar_sample_data.channel, max_intensity=255
            )

    def _log_lidar(self, pointcloud: npt.NDArray[np.float64], channel: str, max_intensity: int) -> None:
        intensity = pointcloud[:, 3] / max_intensity
        points = pointcloud[:, :3]
        cmap = mpl.colormaps["winter_r"]
        point_colors = cmap(intensity)
        rr.log(f"world/ego_vehicle/{channel}", rr.Points3D(points, colors=point_colors))

    def _log_ego_pose(self, darts: DARTS, lidar_sample_data: SampleData) -> None:
        ego_pose = darts.ego_pose.get(lidar_sample_data.ego_pose_token)
        rr.log(
            "world/ego_vehicle",
            rr.Transform3D(
                translation=ego_pose.translation, rotation=rr.Quaternion(xyzw=np.roll(ego_pose.rotation, -1))
            ),
        )

    def _log_sensor_calibration(self, darts: DARTS, sample_data: SampleData) -> None:
        calibration_sensor = darts.calibrated_sensor.get(sample_data.calibrated_sensor_token)
        rr.log(
            f"world/ego_vehicle/{sample_data.channel}",
            rr.Transform3D(
                translation=calibration_sensor.translation,
                rotation=rr.Quaternion(xyzw=np.roll(calibration_sensor.rotation, -1)),
            ),
            static=True,
        )
        if "CAM" in sample_data.channel:
            rr.log(
                f"world/ego_vehicle/{sample_data.channel}",
                rr.Pinhole(
                    image_from_camera=calibration_sensor.camera_intrinsic,
                    width=sample_data.width,
                    height=sample_data.height,
                ),
                static=True,
            )

    def _log_annotations(self, darts: DARTS, annotations: list[SampleAnnotation]) -> None:
        # Rerun expects x=right, y=forward, z=up; dataset uses x=forward, y=left
        rr.log(
            "world/annotations",
            rr.Boxes3D(
                sizes=[[annotation.size[1], annotation.size[0], annotation.size[2]] for annotation in annotations],
                centers=[annotation.translation for annotation in annotations],
                rotations=[rr.Quaternion(xyzw=np.roll(annotation.rotation, -1)) for annotation in annotations],
                labels=[darts.get_category_from_annotation(annotation.token).name for annotation in annotations],
            ),
        )

    def _get_keyframe_sample_data(self, darts: DARTS, sample: Sample, sensor_modality: str) -> list[SampleData]:
        sample_datas: list[SampleData] = []
        for sample_data_token in sample.data.values():
            sample_data = darts.sample_data.get(sample_data_token)
            if sample_data.modality == sensor_modality:
                sample_datas.append(sample_data)
        return sample_datas

    def _get_lidar_pointclouds(self, darts: DARTS, sample_datas: list[SampleData]) -> dict[str, LidarPointCloud]:
        return {sample_data.token: darts.get_lidar_pointcloud(sample_data) for sample_data in sample_datas}
