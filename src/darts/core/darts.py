"""Module for DARTS main class."""

import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from darts.core.table import T, Table
from darts.core.tables import (
    Attribute,
    CalibratedSensor,
    Category,
    EgoPose,
    Ins,
    Instance,
    Instance2D,
    Metadata,
    Sample,
    SampleAnnotation,
    SampleAnnotation2D,
    SampleData,
    Scene,
    Sensor,
    _str_tuple,
)

logger = logging.getLogger(__name__)


class DARTS:
    """Main DARTS class for managing dataset tables.

    Args:
        root: Path to the dataset root directory.
        version: Version of the dataset to load.
        verbose: Whether to show more logs while loading dataset.
        verify_integrity: Whether to check files checksums
    """

    def __init__(self, root: str | Path, version: str, verbose: bool = False, verify_integrity: bool = False) -> None:
        """Initialize the DARTS dataset interface."""
        self._root = Path(root)
        self._version = version
        self._verbose = verbose

        self._attribute = self._load_table("attribute", Attribute)
        self._calibrated_sensor = self._load_table("calibrated_sensor", CalibratedSensor)
        self._category = self._load_table("category", Category)
        self._ego_pose = self._load_table("ego_pose", EgoPose)
        self._ins = self._load_table("ins", Ins)
        self._instance = self._load_table("instance", Instance)
        self._instance_2d = self._load_table("instance_2d", Instance2D)
        self._metadata = self._load_table("metadata", Metadata)
        self._sample = self._load_table("sample", Sample)
        self._sample_annotation = self._load_table("sample_annotation", SampleAnnotation)
        self._sample_annotation_2d = self._load_table("sample_annotation_2d", SampleAnnotation2D)
        self._sample_data = self._load_table("sample_data", SampleData)
        self._scene = self._load_table("scene", Scene)
        self._sensor = self._load_table("sensor", Sensor)

        if verify_integrity:
            self._verify_integrity()

        self._create_relationships()

    @property
    def attribute(self) -> Table[Attribute]:
        """Table of Attribute records."""
        return self._attribute

    @property
    def calibrated_sensor(self) -> Table[CalibratedSensor]:
        """Table of CalibratedSensor records."""
        return self._calibrated_sensor

    @property
    def category(self) -> Table[Category]:
        """Table of Category records."""
        return self._category

    @property
    def ego_pose(self) -> Table[EgoPose]:
        """Table of EgoPose records."""
        return self._ego_pose

    @property
    def ins(self) -> Table[Ins]:
        """Table of Ins records."""
        return self._ins

    @property
    def instance(self) -> Table[Instance]:
        """Table of Instance records."""
        return self._instance

    @property
    def instance_2d(self) -> Table[Instance2D]:
        """Table of Instance2D records."""
        return self._instance_2d

    @property
    def metadata(self) -> Table[Metadata]:
        """Table of Metadata records."""
        return self._metadata

    @property
    def sample(self) -> Table[Sample]:
        """Table of Sample records."""
        return self._sample

    @property
    def sample_annotation(self) -> Table[SampleAnnotation]:
        """Table of SampleAnnotation records."""
        return self._sample_annotation

    @property
    def sample_annotation_2d(self) -> Table[SampleAnnotation2D]:
        """Table of SampleAnnotation2D records."""
        return self._sample_annotation_2d

    @property
    def sample_data(self) -> Table[SampleData]:
        """Table of SampleData records."""
        return self._sample_data

    @property
    def scene(self) -> Table[Scene]:
        """Table of Scene records."""
        return self._scene

    def __repr__(self) -> str:
        """Return a string representation of the DARTS class."""
        return f"DARTS(root={self._root}, version={self._version})"

    def _load_table(self, name: str, cls: type[T]) -> Table[T]:
        if self._verbose:
            logger.info("Loading %s table.", name)
        path = self._root / self._version / f"{name}.json"
        data = json.loads(path.read_text())
        iterable = tqdm(data, desc=f"Loading {name} records", unit="records") if self._verbose else data
        records = [cls.from_dict(item) for item in iterable]
        return Table(records, verbose=self._verbose)

    def _get_checksum(self, filename: str) -> str:
        return hashlib.md5((self._root / filename).read_bytes()).hexdigest()

    def _verify_integrity(self) -> None:
        if self._verbose:
            logger.info("Verifying data integrity")
        sample_data_list = self._sample_data.all()
        iterable = (
            tqdm(sample_data_list, desc="Verifying data integrity", unit="records")
            if self._verbose
            else sample_data_list
        )
        for sample_data in iterable:
            checksum = self._get_checksum(sample_data.filename)
            if checksum != sample_data.checksum:
                msg = f"File {sample_data.filename} has wrong checksum"
                raise ValueError(msg)

    def _add_channel_to_sample_data(self) -> None:
        if self._verbose:
            logger.info("Adding channel to sample data")
        sample_data_list = self._sample_data.all()
        iterable = (
            tqdm(sample_data_list, desc="Adding channel to sample data", unit="records")
            if self._verbose
            else sample_data_list
        )
        for sample_data in iterable:
            calibrated_sensor = self._calibrated_sensor.get(sample_data.calibrated_sensor_token)
            sensor = self._sensor.get(calibrated_sensor.sensor_token)
            sample_data.channel = sensor.channel

    def _add_ins_to_sample(self) -> None:
        if self._verbose:
            logger.info("Adding ins token to sample")
        ins_list = self._ins.all()
        iterable = tqdm(ins_list, desc="Adding ins token to sample", unit="records") if self._verbose else ins_list
        for ins in iterable:
            if not ins.is_key_frame:
                continue
            self._sample.get(ins.sample_token).ins_token = ins.token

    def _add_metadata_to_scene(self) -> None:
        if self._verbose:
            logger.info("Adding metadata token to scene")
        metadata_list = self._metadata.all()
        iterable = (
            tqdm(metadata_list, desc="Adding metadata token to scene", unit="records")
            if self._verbose
            else metadata_list
        )
        for metadata in iterable:
            self._scene.get(metadata.scene_token).metadata_token = metadata.token

    def _add_data_to_sample(self) -> None:
        if self._verbose:
            logger.info("Adding data tokens to sample")
        sample_data_list = self._sample_data.all()
        sample_data_by_sample: defaultdict[str, dict[str, str]] = defaultdict(dict)
        sample_data_iterable = (
            tqdm(sample_data_list, desc="Preparing data tokens for sample", unit="records")
            if self._verbose
            else sample_data_list
        )
        for sample_data in sample_data_iterable:
            if not sample_data.is_key_frame:
                continue
            sample_data_by_sample[sample_data.sample_token][sample_data.channel] = sample_data.token
        sample_list = self._sample.all()
        sample_iterable = (
            tqdm(sample_list, desc="Adding data tokens for sample", unit="records") if self._verbose else sample_list
        )
        for sample in sample_iterable:
            sample.data = sample_data_by_sample.get(sample.token, {})

    def _add_annotations_to_sample(self) -> None:
        if self._verbose:
            logger.info("Adding annotation tokens to sample")
        sample_annotation_list = self._sample_annotation.all()
        sample_annotation_by_sample: defaultdict[str, list[str]] = defaultdict(list)
        sample_annotation_iterable = (
            tqdm(sample_annotation_list, desc="Preparing annotation tokens for sample", unit="records")
            if self._verbose
            else sample_annotation_list
        )
        for sample_annotation in sample_annotation_iterable:
            sample_annotation_by_sample[sample_annotation.sample_token].append(sample_annotation.token)
        sample_list = self._sample.all()
        sample_iterable = (
            tqdm(sample_list, desc="Adding annotation tokens for sample", unit="records")
            if self._verbose
            else sample_list
        )
        for sample in sample_iterable:
            sample.anns = _str_tuple(sample_annotation_by_sample.get(sample.token, []))

    def _add_annotations_to_sample_data(self) -> None:
        if self._verbose:
            logger.info("Adding annotation tokens to camera sample data")
        sample_annotation_2d_list = self._sample_annotation_2d.all()
        sample_annotation_by_sample_data: defaultdict[str, list[str]] = defaultdict(list)
        sample_annotation_2d_iterable = (
            tqdm(sample_annotation_2d_list, desc="Preparing annotation tokens for camera sample data", unit="records")
            if self._verbose
            else sample_annotation_2d_list
        )
        for sample_annotation_2d in sample_annotation_2d_iterable:
            sample = self._sample.get(sample_annotation_2d.sample_token)
            for sample_data_token in sample.data.values():
                sample_data = self._sample_data.get(sample_data_token)
                if sample_data.calibrated_sensor_token == sample_annotation_2d.calibrated_sensor_token:
                    sample_annotation_by_sample_data[sample_data.token].append(sample_annotation_2d.token)
                    break
        sample_data_list = self._sample_data.all()
        sample_data_iterable = (
            tqdm(sample_data_list, desc="Adding annotation tokens to camera sample data", unit="records")
            if self._verbose
            else sample_data_list
        )
        for sample_data in sample_data_iterable:
            sample_data.anns = _str_tuple(sample_annotation_by_sample_data.get(sample_data.token, []))

    def _create_relationships(self) -> None:
        if self._verbose:
            logger.info("Creating relationships between tables")
        self._add_channel_to_sample_data()
        self._add_metadata_to_scene()
        self._add_ins_to_sample()
        self._add_data_to_sample()
        self._add_annotations_to_sample()
        self._add_annotations_to_sample_data()
