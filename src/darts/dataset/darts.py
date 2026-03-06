"""Module for DARTS main class."""

import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from .dataset_models import (
    INS,
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

    def _add_channel_to_sample_data(self) -> None:
        desc = "Adding channel to sample data"
        logger.info(desc)
        iterable = self._progress(self._sample_data.all(), desc, "records")
        for sample_data in iterable:
            calibrated_sensor = self._calibrated_sensor.get(sample_data.calibrated_sensor_token)
            sensor = self._sensor.get(calibrated_sensor.sensor_token)
            sample_data.channel = sensor.channel

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
        sample_data_by_sample: defaultdict[str, dict[str, str]] = defaultdict(dict)
        sample_data_iterable = self._progress(self._sample_data.all(), prepare_desc, "records")

        for sample_data in sample_data_iterable:
            if not sample_data.is_key_frame:
                continue
            sample_data_by_sample[sample_data.sample_token][sample_data.channel] = sample_data.token
        sample_iterable = self._progress(self._sample.all(), adding_desc, "records")
        for sample in sample_iterable:
            sample.data = sample_data_by_sample.get(sample.token, {})

    def _add_annotations_to_sample(self) -> None:
        adding_desc = "Adding annotation tokens to sample"
        prepare_desc = "Preparing annotation tokens for sample"
        logger.info(adding_desc)
        sample_annotation_by_sample: defaultdict[str, list[str]] = defaultdict(list)
        sample_annotation_iterable = self._progress(self._sample_annotation.all(), prepare_desc, "records")

        for sample_annotation in sample_annotation_iterable:
            sample_annotation_by_sample[sample_annotation.sample_token].append(sample_annotation.token)
        sample_iterable = self._progress(self._sample.all(), adding_desc, "records")
        for sample in sample_iterable:
            sample.anns = str_tuple(sample_annotation_by_sample.get(sample.token, []))

    def _add_annotations_to_sample_data(self) -> None:
        adding_desc = "Adding annotation tokens to camera sample data"
        prepare_desc = "Preparing annotation tokens for camera sample data"
        logger.info(adding_desc)
        sample_annotation_by_sample_data: defaultdict[str, list[str]] = defaultdict(list)
        sample_annotation_2d_iterable = self._progress(self._sample_annotation_2d.all(), prepare_desc, "records")

        for sample_annotation_2d in sample_annotation_2d_iterable:
            sample = self._sample.get(sample_annotation_2d.sample_token)
            for sample_data_token in sample.data.values():
                sample_data = self._sample_data.get(sample_data_token)
                if sample_data.calibrated_sensor_token == sample_annotation_2d.calibrated_sensor_token:
                    sample_annotation_by_sample_data[sample_data.token].append(sample_annotation_2d.token)
                    break

        sample_data_iterable = self._progress(self._sample_data.all(), adding_desc, "records")
        for sample_data in sample_data_iterable:
            sample_data.anns = str_tuple(sample_annotation_by_sample_data.get(sample_data.token, []))

    def _create_relationships(self) -> None:
        logger.info("Creating relationships between tables")
        self._add_channel_to_sample_data()
        self._add_scene_metadata_to_scene()
        self._add_ins_to_sample()
        self._add_data_to_sample()
        self._add_annotations_to_sample()
        self._add_annotations_to_sample_data()
