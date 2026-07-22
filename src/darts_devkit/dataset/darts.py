"""Module for DARTS main class."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import TypeAlias

import PIL.Image
import PIL.ImageFile
from tqdm import tqdm

from .data_classes import LidarPointCloud
from .dataset_models import (
    INS,
    Attribute,
    CalibratedSensor,
    Category,
    EgoPose,
    Instance,
    Instance2D,
    Sample,
    SampleAnnotation,
    SampleAnnotation2D,
    SampleData,
    Scene,
    SceneMetadata,
    Sensor,
    str_tuple,
)
from .record_collection import RecordCollection, T

logger = logging.getLogger(__name__)

MetadataValue: TypeAlias = str | int | float | bool | tuple[str] | tuple[int] | tuple[float] | tuple[bool]

QueryValue: TypeAlias = str | int | float | bool | list[str] | list[int] | list[float] | list[bool]


class DARTS:
    """Main DARTS class for managing dataset tables.

    Args:
        root: Path to the dataset root directory.
        version: Version of the dataset to load.
    """

    def __init__(self, root: str | Path, version: str) -> None:
        """Initialize the DARTS dataset interface."""
        self._root = Path(root)
        self._version = version
        self._show_progress = sys.stderr.isatty() and logger.isEnabledFor(logging.INFO)

        self._calibrated_sensor = self._load_table("calibrated_sensor", CalibratedSensor)
        self._category = self._load_table("category", Category)
        self._ego_pose = self._load_table("ego_pose", EgoPose)
        self._ins = self._load_table("ins", INS)
        self._instance = self._load_table("instance", Instance)
        self._instance_2d = self._load_table("instance_2d", Instance2D)
        self._scene_metadata = self._load_table("metadata", SceneMetadata)
        self._sample = self._load_table("sample", Sample)
        self._sample_annotation = self._load_table("sample_annotation", SampleAnnotation)
        self._sample_annotation_2d = self._load_table("sample_annotation_2d", SampleAnnotation2D)
        self._sample_data = self._load_table("sample_data", SampleData)
        self._scene = self._load_table("scene", Scene)
        self._sensor = self._load_table("sensor", Sensor)
        self._attribute = self._load_table("attribute", Attribute)

        self._create_relationships()

    @property
    def calibrated_sensor(self) -> RecordCollection[CalibratedSensor]:
        """RecordCollection of CalibratedSensor records."""
        return self._calibrated_sensor

    @property
    def category(self) -> RecordCollection[Category]:
        """RecordCollection of Category records."""
        return self._category

    @property
    def attribute(self) -> RecordCollection[Attribute]:
        """RecordCollection of Attribute records."""
        return self._attribute

    @property
    def sensor(self) -> RecordCollection[Sensor]:
        """RecordCollection of Sensor records."""
        return self._sensor

    @property
    def ego_pose(self) -> RecordCollection[EgoPose]:
        """RecordCollection of EgoPose records."""
        return self._ego_pose

    @property
    def ins(self) -> RecordCollection[INS]:
        """RecordCollection of Ins records."""
        return self._ins

    @property
    def instance(self) -> RecordCollection[Instance]:
        """RecordCollection of Instance records."""
        return self._instance

    @property
    def instance_2d(self) -> RecordCollection[Instance2D]:
        """RecordCollection of Instance2D records."""
        return self._instance_2d

    @property
    def scene_metadata(self) -> RecordCollection[SceneMetadata]:
        """RecordCollection of SceneMetadata records."""
        return self._scene_metadata

    @property
    def sample(self) -> RecordCollection[Sample]:
        """RecordCollection of Sample records."""
        return self._sample

    @property
    def sample_annotation(self) -> RecordCollection[SampleAnnotation]:
        """RecordCollection of SampleAnnotation records."""
        return self._sample_annotation

    @property
    def sample_annotation_2d(self) -> RecordCollection[SampleAnnotation2D]:
        """RecordCollection of SampleAnnotation2D records."""
        return self._sample_annotation_2d

    @property
    def sample_data(self) -> RecordCollection[SampleData]:
        """RecordCollection of SampleData records."""
        return self._sample_data

    @property
    def scene(self) -> RecordCollection[Scene]:
        """RecordCollection of Scene records."""
        return self._scene

    def verify_integrity(self) -> None:
        """This method checks checksums of sample_data files.

        :raises ValueError: If some file has bad checksum
        """
        desc = "Verifying data integrity"
        bad_files_list = []
        logger.info(desc)
        iterable = self._progress(self._sample_data.all(), desc, "records")
        for sample_data in iterable:
            checksum = self._get_checksum(sample_data.filename)
            if checksum != sample_data.checksum:
                bad_files_list.append(sample_data.filename)

        if len(bad_files_list) > 0:
            msg = "Some files have wrong checksums: \n"
            for bad_file in bad_files_list:
                msg += f"{bad_file}\n"
            logger.error(msg)
            raise ValueError(msg)

    def filter_scenes(self, query: dict) -> DARTS:
        """Returs DARTS instance with filtered scene based on query with possible NOT, AND, OR operands.

        Args:
            query: logical query with which we filter records
        Returns:
            DARTS instance with filtered records

        Example:
        --------
        .. code-block:: python

            query = {
                "or": [
                    {"not": {"intersection_y": 1}},
                    {"traffic_participants": ["trucks", "cyclist"]},
                ]
            }


        A list is treated as a disjunction, not a conjunction. It means that at least one of the values of the list
        must be present in the scene metadata. In the example above, a track or a cyclist should be present in the
        recording, not both of them.
        """
        metadata = [m for m in self._scene_metadata.all() if self._match_query(m, query)]

        scene_tokens = {m.scene_token for m in metadata}
        scenes = self._collect(self._scene, scene_tokens)

        samples = self._samples_from_scenes(scenes)
        sample_datas = self._sample_data_from_samples(samples)

        sample_annotations = self.get_annotations_from_samples(samples)
        sample_annotations_2d = self.get_annotations_2d_from_sample_datas(sample_datas)
        ins = self._ins_from_samples(samples)

        calibrated_sensors = self._collect(
            self._calibrated_sensor,
            {s.calibrated_sensor_token for s in sample_datas},
        )

        ego_poses = self._collect(
            self._ego_pose,
            {s.ego_pose_token for s in sample_datas},
        )

        instances = self._collect(
            self._instance,
            {a.instance_token for a in sample_annotations},
        )

        instances_2d = self._collect(
            self._instance_2d,
            {a.instance_2d_token for a in sample_annotations_2d},
        )

        filtered = self._clone_empty()
        filtered.__dict__.update(
            {
                "_scene_metadata": RecordCollection(metadata),
                "_scene": RecordCollection(scenes),
                "_sample": RecordCollection(samples),
                "_sample_data": RecordCollection(sample_datas),
                "_sample_annotation": RecordCollection(sample_annotations),
                "_sample_annotation_2d": RecordCollection(sample_annotations_2d),
                "_ins": RecordCollection(ins),
                "_instance": RecordCollection(instances),
                "_instance_2d": RecordCollection(instances_2d),
                "_ego_pose": RecordCollection(ego_poses),
                "_calibrated_sensor": RecordCollection(calibrated_sensors),
                "_category": self._category,
                "_sensor": self._sensor,
            }
        )

        return filtered

    def get_annotations_from_samples(self, samples: list[Sample]) -> list[SampleAnnotation]:
        """Get annotations from list of samples.

        Args:
            samples: list of Samples from which to return 3D annotations
        Returns:
            list of SampleAnnotation object instances
        """
        anns: list[SampleAnnotation] = []
        for sample in samples:
            anns.extend(self._sample_annotation.get(token) for token in sample.anns)
        return anns

    def get_annotations_2d_from_sample_datas(self, sample_datas: list[SampleData]) -> list[SampleAnnotation2D]:
        """Get 2D annotations from list of camera sample datas.

        Args:
            sample_datas: list of SampleData from which to return 2D annotations
        Returns:
            list of SampleAnnotation2D object instances
        """
        anns: list[SampleAnnotation2D] = []
        for sample_data in sample_datas:
            if sample_data.modality != "camera":
                continue
            anns.extend(self._sample_annotation_2d.get(token) for token in sample_data.anns)
        return anns

    def get_lidar_pointcloud(self, sample_data: SampleData) -> LidarPointCloud:
        """Get lidar pointcloud from sample data with lidar modality.

        Args:
            sample_data: SampleData from which to return LidarPointCloud
        Returns:
            LidarPointCloud instance
        """
        if sample_data.modality != "lidar":
            msg = f"Sample data is not of lidar file but {sample_data.modality}"
            logger.error(msg)
            raise TypeError(msg)
        return LidarPointCloud.from_file(self._get_sensor_data_path(sample_data).as_posix())

    def get_image(self, sample_data: SampleData) -> PIL.ImageFile.ImageFile:
        """Returns image for given camera sample.

        Args:
            sample_data: camera sample data
        Returns:
            image for given camera sample
        """
        return PIL.Image.open(self._get_sensor_data_path(sample_data))

    def get_samples_from_scene(self, scene_token: str) -> list[Sample]:
        """Get samples from scene.

        Args:
            scene_token: token of scene to get samples from
        Returns:
            list of Sample object instances
        """
        scene = self._scene.get(scene_token)
        samples = [self._sample.get(scene.first_sample_token)]
        while samples[-1].next != "":
            samples.append(self._sample.get(samples[-1].next))
        return samples

    def get_sensor_from_annotation_2d(self, annotation_2d_token: str) -> Sensor:
        """Get sensor from 2D annotation.

        Args:
            annotation_2d_token: token of 2D annotation to get sensor from
        Returns:
            Sensor object instance
        """
        annotation_2d = self._sample_annotation_2d.get(annotation_2d_token)
        calibrated_sensor = self._calibrated_sensor.get(annotation_2d.calibrated_sensor_token)
        return self._sensor.get(calibrated_sensor.sensor_token)

    def get_category_from_annotation_2d(self, annotation_2d_token: str) -> Category:
        """Get category from 2D annotation.

        Args:
            annotation_2d_token: token of 2D annotation to get category from
        Returns:
            Category object instance
        """
        annotation_2d = self._sample_annotation_2d.get(annotation_2d_token)
        instance_2d = self._instance_2d.get(annotation_2d.instance_2d_token)
        return self._category.get(instance_2d.category_token)

    def get_category_from_annotation(self, annotation_token: str) -> Category:
        """Get category from 3D annotation.

        Args:
            annotation_token: token of 3D annotation to get category from
        Returns:
            Category object instance
        """
        annotation = self._sample_annotation.get(annotation_token)
        instance = self._instance.get(annotation.instance_token)
        return self._category.get(instance.category_token)

    def _get_sensor_data_path(self, sample_data: SampleData) -> Path:
        return Path(self._root) / str(sample_data.filename)

    def _get_sample_datas_from_sample(self, sample_token: str) -> list[SampleData]:
        sample = self._sample.get(sample_token)
        sample_datas: list[SampleData] = []
        for sample_data_token in sample.data.values():
            sample_datas_temp = [self._sample_data.get(sample_data_token)]
            while (
                sample_datas_temp[-1].prev != ""
                and self._sample_data.get(sample_datas_temp[-1].prev).sample_token == sample_token
            ):
                sample_datas_temp.append(self._sample_data.get(sample_datas_temp[-1].prev))

            sample_datas_temp = [*sample_datas_temp[1:], sample_datas_temp[0]]
            while (
                sample_datas_temp[-1].next != ""
                and self._sample_data.get(sample_datas_temp[-1].next).sample_token == sample_token
            ):
                sample_datas_temp.append(self._sample_data.get(sample_datas_temp[-1].next))
            sample_datas += sample_datas_temp
        return sample_datas

    def _get_ins_from_sample(self, sample_token: str) -> list[INS]:
        sample = self._sample.get(sample_token)
        ins_data: list[INS] = [self._ins.get(sample.ins_token)]
        while ins_data[-1].prev != "" and self._ins.get(ins_data[-1].prev).sample_token == sample_token:
            ins_data.append(self._ins.get(ins_data[-1].prev))

        ins_data = [*ins_data[1:], ins_data[0]]
        while ins_data[-1].next != "" and self._ins.get(ins_data[-1].next).sample_token == sample_token:
            ins_data.append(self._ins.get(ins_data[-1].next))
        return ins_data

    def _collect(self, collection: RecordCollection[T], tokens: set[str]) -> list[T]:
        return [collection.get(t) for t in tokens]

    def _samples_from_scenes(self, scenes: list[Scene]) -> list[Sample]:
        samples = []
        for scene in scenes:
            samples += self.get_samples_from_scene(scene.token)
        return samples

    def _sample_data_from_samples(self, samples: list[Sample]) -> list[SampleData]:
        result = []
        for sample in samples:
            result += self._get_sample_datas_from_sample(sample.token)
        return result

    def _ins_from_samples(self, samples: list[Sample]) -> list[INS]:
        result = []
        for sample in samples:
            result += self._get_ins_from_sample(sample.token)
        return result

    def _match_query(self, metadata: SceneMetadata, query: dict) -> bool:
        if not isinstance(query, dict):
            msg = f"Query {query} is not a dictionary."
            logger.error(msg)
            raise KeyError(msg)
        if len(query.keys()) != 1:
            msg = f"Query {query} does not have one key."
            logger.error(msg)
            raise KeyError(msg)
        if "and" in query:
            return self._match_and(metadata, query["and"])

        if "or" in query:
            return self._match_or(metadata, query["or"])

        if "not" in query:
            return self._match_not(metadata, query["not"])

        return self._match_fields(metadata, query)

    def _match_and(self, metadata: SceneMetadata, queries: list[dict]) -> bool:
        return all(self._match_query(metadata, q) for q in queries)

    def _match_or(self, metadata: SceneMetadata, queries: list[dict]) -> bool:
        return any(self._match_query(metadata, q) for q in queries)

    def _match_not(self, metadata: SceneMetadata, query: dict) -> bool:
        return not self._match_query(metadata, query)

    def _match_fields(self, metadata: SceneMetadata, query: dict) -> bool:
        for field, value in query.items():
            metadata_value = getattr(metadata, field)

            if not self._match_field(metadata_value, value):
                return False

        return True

    def _match_field(self, metadata_value: MetadataValue, query_value: QueryValue) -> bool:
        metadata_values = list(metadata_value) if isinstance(metadata_value, tuple) else [metadata_value]
        query_values = query_value if isinstance(query_value, list) else [query_value]

        for mv in metadata_values:
            for qv in query_values:
                if not isinstance(qv, type(mv)):
                    msg = f"Type mismatch: metadata {mv} has {type(mv).__name__}, query {qv} has {type(qv).__name__}"
                    logger.error(msg)
                    raise TypeError(msg)
        return any(v in metadata_values for v in query_values)

    def _clone_empty(self) -> DARTS:
        obj = object.__new__(DARTS)
        obj.__dict__ = self.__dict__.copy()

        for k, v in obj.__dict__.items():
            if isinstance(v, RecordCollection):
                obj.__dict__[k] = RecordCollection([])

        return obj

    def __repr__(self) -> str:
        """Return a string representation of the DARTS class."""
        return f"DARTS(root={self._root}, version={self._version})"

    def _progress(self, iterable: list[T], desc: str, unit: str) -> tqdm[T] | list[T]:
        if self._show_progress:
            return tqdm(iterable, desc=desc, unit=unit)
        return iterable

    def _load_table(self, name: str, cls: type[T]) -> RecordCollection[T]:
        logger.info("Loading %s table.", name)
        path = self._root / self._version / f"{name}.json"
        data = json.loads(path.read_text())
        iterable = tqdm(data, desc=f"Loading {name} records", unit="records") if self._show_progress else data
        records = [cls.from_dict(item) for item in iterable]
        return RecordCollection(records)

    def _get_checksum(self, filename: str) -> str:
        return hashlib.md5((self._root / filename).read_bytes()).hexdigest()

    def _add_channel_and_modality_to_sample_data(self) -> None:
        desc = "Adding channel to sample data"
        logger.info(desc)
        iterable = self._progress(self._sample_data.all(), desc, "records")
        for sample_data in iterable:
            calibrated_sensor = self._calibrated_sensor.get(sample_data.calibrated_sensor_token)
            sensor = self._sensor.get(calibrated_sensor.sensor_token)
            sample_data.channel = sensor.channel
            sample_data.modality = sensor.modality

    def _add_ins_to_sample(self) -> None:
        desc = "Adding ins token to sample"
        logger.info(desc)
        iterable = self._progress(self._ins.all(), desc, "records")
        for ins in iterable:
            if not ins.is_key_frame:
                continue
            self._sample.get(ins.sample_token).ins_token = ins.token

    def _add_scene_metadata_to_scene(self) -> None:
        desc = "Adding scene metadata token to scene"
        logger.info(desc)
        iterable = self._progress(self._scene_metadata.all(), desc, "records")
        for scene_metadata in iterable:
            self._scene.get(scene_metadata.scene_token).scene_metadata_token = scene_metadata.token

    def _add_data_to_sample(self) -> None:
        adding_desc = "Adding data tokens to sample"
        prepare_desc = "Preparing data tokens for sample"
        logger.info(adding_desc)
        sample_data_from_sample: dict[str, dict[str, str]] = defaultdict(dict)
        sample_data_iterable = self._progress(self._sample_data.all(), prepare_desc, "records")

        for sample_data in sample_data_iterable:
            if not sample_data.is_key_frame:
                continue
            sample_data_from_sample[sample_data.sample_token][sample_data.channel] = sample_data.token
        sample_iterable = self._progress(self._sample.all(), adding_desc, "records")
        for sample in sample_iterable:
            sample.data = sample_data_from_sample.get(sample.token, {})

    def _add_annotations_to_sample(self) -> None:
        adding_desc = "Adding annotation tokens to sample"
        prepare_desc = "Preparing annotation tokens for sample"
        logger.info(adding_desc)
        sample_annotation_from_sample: dict[str, list[str]] = defaultdict(list)
        sample_annotation_iterable = self._progress(self._sample_annotation.all(), prepare_desc, "records")

        for sample_annotation in sample_annotation_iterable:
            sample_annotation_from_sample[sample_annotation.sample_token].append(sample_annotation.token)
        sample_iterable = self._progress(self._sample.all(), adding_desc, "records")
        for sample in sample_iterable:
            sample.anns = str_tuple(sample_annotation_from_sample.get(sample.token, []))

    def _add_annotations_to_sample_data(self) -> None:
        adding_desc = "Adding annotation tokens to camera sample data"
        prepare_desc = "Preparing annotation tokens for camera sample data"
        logger.info(adding_desc)
        sample_annotation_from_sample_data: dict[str, list[str]] = defaultdict(list)
        sample_annotation_2d_iterable = self._progress(self._sample_annotation_2d.all(), prepare_desc, "records")

        for sample_annotation_2d in sample_annotation_2d_iterable:
            sample = self._sample.get(sample_annotation_2d.sample_token)
            for sample_data_token in sample.data.values():
                sample_data = self._sample_data.get(sample_data_token)
                if sample_data.calibrated_sensor_token == sample_annotation_2d.calibrated_sensor_token:
                    sample_annotation_from_sample_data[sample_data.token].append(sample_annotation_2d.token)
                    break

        sample_data_iterable = self._progress(self._sample_data.all(), adding_desc, "records")
        for sample_data in sample_data_iterable:
            sample_data.anns = str_tuple(sample_annotation_from_sample_data.get(sample_data.token, []))

    def _create_relationships(self) -> None:
        logger.info("Creating relationships between tables")
        self._add_channel_and_modality_to_sample_data()
        self._add_scene_metadata_to_scene()
        self._add_ins_to_sample()
        self._add_data_to_sample()
        self._add_annotations_to_sample()
        self._add_annotations_to_sample_data()
