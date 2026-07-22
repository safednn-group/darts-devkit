"""Dataset tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypeAlias

from .record_collection import Record

Vec3: TypeAlias = tuple[float, float, float]
"""`Vec3` is a type variable representing any x,y,z vector."""

Corners: TypeAlias = tuple[float, float, float, float]
"""`Corners` is a type variable representing any min_x, min_y, max_x, max_y values."""

Quaternion: TypeAlias = tuple[float, float, float, float]
"""`Quaternion` is a type variable representing any w,x,y,z Quaternion."""

Mat3: TypeAlias = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
"""`Mat3` is a type variable representing 3x3 matrix."""

StrTuple: TypeAlias = tuple[str, ...]
"""Immutable sequence of strings."""

IntTuple: TypeAlias = tuple[int, ...]
"""Immutable sequence of integers."""


def vec3(values: list[float]) -> Vec3:
    """Construct a variable representing any x,y,z vector.

    Args:
        values: A 3 element list of floats

    Returns:
        tuple[x, y, z]

    """
    try:
        x, y, z = values
    except ValueError as exc:
        msg = "Expected 3 elements"
        raise ValueError(msg) from exc
    return (x, y, z)


def corners(values: list[float]) -> Corners:
    """Construct a type variable representing any min_x, min_y, max_x, max_y values.

    Args:
        values: A 4 element list of floats

    Returns:
        tuple[min_x, min_y, max_x, max_y]

    """
    try:
        min_x, min_y, max_x, max_y = values
    except ValueError as exc:
        msg = "Expected 4 elements"
        raise ValueError(msg) from exc
    return (min_x, min_y, max_x, max_y)


def quat(values: list[float]) -> Quaternion:
    """Construct a type variable representing any w,x,y,z Quaternion.

    Args:
        values: A 4 element list of floats

    Returns:
        tuple[w, x, y, z]

    """
    try:
        w, x, y, z = values
    except ValueError as exc:
        msg = "Expected 4 elements"
        raise ValueError(msg) from exc
    return (w, x, y, z)


def mat3(values: list[list[float]] | None) -> Mat3 | None:
    """Construct a type variable representing 3x3 matrix.

    Args:
        values: optional A 3x3 element list of floats

    Returns:
        tuple[tuple[x_1, x_2, x_3], tuple[x_4, x_5, x_6], tuple[x_7, x_8, x_9]] if values was not None, else None

    """
    if not values:
        return None
    try:
        x_1, x_2, x_3 = values[0]
        x_4, x_5, x_6 = values[1]
        x_7, x_8, x_9 = values[2]
    except ValueError as exc:
        msg = "Expected 3x3 elements"
        raise ValueError(msg) from exc
    return ((x_1, x_2, x_3), (x_4, x_5, x_6), (x_7, x_8, x_9))


def str_tuple(values: list[str]) -> StrTuple:
    """Construct Tuple of strings from list.

    Args:
        values: A list of strings

    Returns:
        tuple[str, ...]

    """
    for value in values:
        if not isinstance(value, str):
            msg = "Value is not str"
            raise TypeError(msg)
    return tuple(values)


def int_tuple(values: list[int]) -> IntTuple:
    """Construct Tuple of ints from list.

    Args:
        values: A list of ints

    Returns:
        tuple[int, ...]

    """
    for value in values:
        if not isinstance(value, int):
            msg = "Value is not int"
            raise TypeError(msg)
    return tuple(values)


@dataclass(slots=True)
class Timestamp:
    """Immutable timestamp wrapper.

    Stores a Unix timestamp in microseconds and provides convenient
    access to a timezone-aware UTC datetime.

    Attributes:
        timestamp: Unix timestamp in microseconds.
    """

    timestamp: int

    @property
    def as_datetime(self) -> datetime:
        """Return the timestamp as a UTC datetime.

        Returns:
            datetime: A timezone-aware datetime in UTC corresponding to the timestamp.
        """
        return datetime.fromtimestamp(self.timestamp / 1_000_000, tz=timezone.utc)


@dataclass(slots=True)
class CalibratedSensor(Record):
    """Calibration parameters for a specific sensor instance.

    Represents the rigid transformation from the ego vehicle frame
    to the sensor frame.

    Attributes:
        token: Unique identifier of the record.
        sensor_token: Reference to the associated sensor.
        translation: 3D translation vector in meters.
        rotation: Rotation as a quaternion.
        camera_intrinsic: Camera intrinsic matrix or None for non-camera sensors.
    """

    token: str
    sensor_token: str
    translation: Vec3
    rotation: Quaternion
    camera_intrinsic: Mat3 | None = None

    @classmethod
    def from_dict(cls, data: dict) -> CalibratedSensor:
        """Construct an instance from a dictionary.

        Args:
            data: A dictionary containing the record fields.

        Returns:
            An instance of the subclass.

        """
        return cls(
            translation=vec3(data.pop("translation")),
            rotation=quat(data.pop("rotation")),
            camera_intrinsic=mat3(data.pop("camera_intrinsic")),
            **data,
        )


@dataclass(slots=True)
class Category(Record):
    """Object category definition.

    Attributes:
        token: Unique identifier of the record.
        name: Canonical name of the category.
        description: Detailed explanation of the category.
    """

    token: str
    name: str
    description: str


@dataclass(slots=True)
class EgoPose(Record, Timestamp):
    """Ego vehicle pose at a particular timestamp.

    Pose is expressed with respect to the global coordinate system.

    Attributes:
        token: Unique identifier of the record.
        translation: 3D translation vector in meters.
        rotation: Rotation as a quaternion.
        timestamp: Unix timestamp in microseconds.
    """

    token: str
    translation: Vec3
    rotation: Quaternion

    @classmethod
    def from_dict(cls, data: dict) -> EgoPose:
        """Construct an instance from a dictionary.

        Args:
            data: A dictionary containing the record fields.

        Returns:
            An instance of the subclass.

        """
        return cls(translation=vec3(data.pop("translation")), rotation=quat(data.pop("rotation")), **data)


@dataclass(slots=True)
class INS(Record, Timestamp):
    """Inertial Navigation System (INS) record.

    Represents ego-vehicle pose and motion state at a specific timestamp.
    Each record is linked to a sample and references adjacent INS records
    in the sequence.

    Attributes:
        token: Unique identifier for this INS record.
        sample_token: Token of the associated sample.
        timestamp: Unix timestamp in microseconds.
        is_key_frame: Whether this record corresponds to a key sample.
        prev: Token of the previous INS record in the sequence.
        next: Token of the next INS record in the sequence.

        latitude: Geographic latitude in degrees.
        longitude: Geographic longitude in degrees.
        altitude: Altitude in meters above sea level.

        roll: Roll angle in radians.
        pitch: Pitch angle in radians.
        yaw: Yaw angle in radians.

        velocity_x: Linear velocity along the X axis (m/s).
        velocity_y: Linear velocity along the Y axis (m/s).
        velocity_z: Linear velocity along the Z axis (m/s).

        angular_velocity_body_x: Angular velocity around body-frame X axis (rad/s).
        angular_velocity_body_y: Angular velocity around body-frame Y axis (rad/s).
        angular_velocity_body_z: Angular velocity around body-frame Z axis (rad/s).

        acceleration_x: Linear acceleration along the X axis (m/s²).
        acceleration_y: Linear acceleration along the Y axis (m/s²).
        acceleration_z: Linear acceleration along the Z axis (m/s²).
    """

    token: str
    sample_token: str
    is_key_frame: bool
    prev: str
    next: str
    latitude: float
    longitude: float
    altitude: float
    roll: float
    pitch: float
    yaw: float
    velocity_x: float
    velocity_y: float
    velocity_z: float
    angular_velocity_body_x: float
    angular_velocity_body_y: float
    angular_velocity_body_z: float
    acceleration_x: float
    acceleration_y: float
    acceleration_z: float


@dataclass(slots=True)
class Instance(Record):
    """3D Object instance record.

    Represents a tracked 3D object instance across multiple annotations.
    An instance groups all 3D annotations that belong to the same physical
    object throughout a sequence.

    Attributes:
        token: Unique identifier for this instance.
        category_token: Token referencing the associated category record.
        nbr_annotations: Total number of 3D annotations linked to this 3D instance.
        first_annotation_token: Token of the first 3D annotation in the sequence.
        last_annotation_token: Token of the last 3D annotation in the sequence.
    """

    token: str
    category_token: str
    nbr_annotations: int
    first_annotation_token: str
    last_annotation_token: str


@dataclass(slots=True)
class Instance2D(Record):
    """2D Object instance record.

    Represents a tracked 2D object instance across multiple annotations by camera.
    An instance2D groups all 2D annotations that belong to the same physical
    object throughout a sequence in one camera.

    Attributes:
        token: Unique identifier for this instance.
        category_token: Token referencing the associated category record.
        instance_token: Token referencing the associated instance record.
            Intance2D records of the same object have the same instance token.
            If instance does not have 3D annotations instance record will not be present.
        calibrated_sensor_token: Token referencing the associated calibrated sensor record.
            By this token we can infer which camera recorded this object.
        nbr_annotations: Total number of 2D annotations linked to this 2D instance.
        first_image_annotation_token: Token of the first 2D annotation in the sequence in this camera.
        last_image_annotation_token: Token of the last 2D annotation in the sequence in this camera.
    """

    token: str
    category_token: str
    instance_token: str
    calibrated_sensor_token: str
    nbr_annotations: int
    first_image_annotation_token: str
    last_image_annotation_token: str


@dataclass(slots=True)
class SceneMetadata(Record):
    """Scene-level contextual metadata.

    Represents environmental, infrastructure, traffic, and movement
    information associated with a dataset scene.

    Attributes:
        token: Unique identifier of this record.
        scene_token: Token referencing the associated scene.
        points_category: Points classification category.
        date: Timestamp of the scene capture. (equal to first sample timestamp)

        intersection_y: Whether a Y-intersection exists (0/1).
        intersection_x: Whether an X-intersection exists (0/1).
        intersection_t: Whether a T-intersection exists (0/1).
        roundabout: Whether a roundabout exists (0/1).
        number_of_roadways: Total number of roadways.
        directions: Traffic direction configuration.

        surface: Road surface type.
        surface_weather_condition: Weather condition affecting the surface.
        surface_physical_condition: Physical condition of the surface.
        road_shoulder_left: Description of left road shoulder.
        road_shoulder_right: Description of right road shoulder.
        level_crossing: Level crossing information.
        road_section_type: Type of road section.
        area: Area classification.
        topology: Topology classification.
        time_of_day: Time-of-day category.
        season: Season category.

        vertical_road_markings: Condition/type of vertical road markings.
        horizontal_road_markings: Condition/type of horizontal road markings.

        longitude: Longitude coordinate.
        latitude: Latitude coordinate.

        road_geometry: Road geometry descriptors.
        traffic_infrastructure: Infrastructure elements present.
        road_signs_types: Types of road signs present.
        events: Events occurring in the scene.
        road_category: Road category labels.
        number_of_lanes: Lane counts.
        traffic_participants: Types of traffic participants.
        environment_conditions: Environmental conditions.
        straight_movement: Straight movement behaviors.
        curve_movement: Curve movement behaviors.

        warning_signs: Warning sign identifiers.
        prohibition_signs: Prohibition sign identifiers.
        mandatory_signs: Mandatory sign identifiers.
        information_signs: Information sign identifiers.
        direction_and_location_signs: Direction/location sign identifiers.
        supplementary_signs: Supplementary sign identifiers.
        road_plates_signs: Road plate sign identifiers.
    """

    token: str
    scene_token: str
    points_category: str
    date: datetime
    intersection_y: int
    intersection_x: int
    intersection_t: int
    roundabout: int
    number_of_roadways: int
    directions: str
    surface: str
    surface_weather_condition: str
    surface_physical_condition: str
    road_shoulder_left: str
    road_shoulder_right: str
    level_crossing: str
    road_section_type: str
    area: str
    topology: str
    time_of_day: str
    season: str
    vertical_road_markings: str
    horizontal_road_markings: str
    longitude: float
    latitude: float
    road_geometry: StrTuple
    traffic_infrastructure: StrTuple
    road_signs_types: StrTuple
    events: StrTuple
    road_category: StrTuple
    number_of_lanes: IntTuple
    traffic_participants: StrTuple
    environment_conditions: StrTuple
    straight_movement: StrTuple
    curve_movement: StrTuple
    warning_signs: StrTuple
    prohibition_signs: StrTuple
    mandatory_signs: StrTuple
    information_signs: StrTuple
    direction_and_location_signs: StrTuple
    supplementary_signs: StrTuple
    road_plates_signs: StrTuple

    @classmethod
    def from_dict(cls, data: dict) -> SceneMetadata:
        """Construct an instance from a dictionary.

        Args:
            data: A dictionary containing the record fields.

        Returns:
            An instance of the subclass.

        """
        return cls(
            date=datetime.fromisoformat(data.pop("date")),
            road_geometry=str_tuple(data.pop("road_geometry")),
            traffic_infrastructure=str_tuple(data.pop("traffic_infrastructure")),
            road_signs_types=str_tuple(data.pop("road_signs_types")),
            events=str_tuple(data.pop("events")),
            road_category=str_tuple(data.pop("road_category")),
            number_of_lanes=int_tuple(data.pop("number_of_lanes")),
            traffic_participants=str_tuple(data.pop("traffic_participants")),
            environment_conditions=str_tuple(data.pop("environment_conditions")),
            straight_movement=str_tuple(data.pop("straight_movement")),
            curve_movement=str_tuple(data.pop("curve_movement")),
            warning_signs=str_tuple(data.pop("warning_signs")),
            prohibition_signs=str_tuple(data.pop("prohibition_signs")),
            mandatory_signs=str_tuple(data.pop("mandatory_signs")),
            information_signs=str_tuple(data.pop("information_signs")),
            direction_and_location_signs=str_tuple(data.pop("direction_and_location_signs")),
            supplementary_signs=str_tuple(data.pop("supplementary_signs")),
            road_plates_signs=str_tuple(data.pop("road_plates_signs")),
            **data,
        )


@dataclass(slots=True)
class Sample(Record, Timestamp):
    """Dataset sample record.

    A sample represents a keyframe in a scene and links to sensor data
    and annotations captured at a specific timestamp.

    Attributes:
        token: Unique identifier of this sample.
        timestamp: Unix timestamp in microseconds.
        scene_token: Token referencing the associated scene.
        prev: Token of the previous sample in the scene (empty string if none).
        next: Token of the next sample in the scene (empty string if none).
        data: Dictionary with sample_data tokens by channel
        anns: List of 3D annotations present in this sample
        ins_token: Token referencing the associated ins.
    """

    token: str
    timestamp: int
    scene_token: str
    prev: str
    next: str
    data: dict[str, str] = field(default_factory=dict)
    anns: StrTuple = field(default_factory=tuple)
    ins_token: str = ""


@dataclass(slots=True)
class SampleAnnotation(Record):
    """A single 3D annotation of an object in a sample.

    Represents a 3D bounding box in a sample, associated with an 3D instance
    and optional attributes.

    Attributes:
        token: Unique identifier of the record.
        sample_token: Token referencing the associated sample.
        instance_token: Token referencing the associated instance.
        translation: 3D position (x, y, z) in meters.
        size: 3D size (width, length, height) in meters.
        rotation: Quaternion rotation (w, x, y, z).
        score: Score of annotation.
        prev: Token of the previous annotation ("" if none).
        next: Token of the next annotation ("" if none).
        num_lidar_pts: Number of LiDAR points within the bounding box.
        num_radar_pts: Number of radar points within the bounding box.
        attribute_tokens: Tuple of associated attribute tokens.
    """

    token: str
    sample_token: str
    instance_token: str
    translation: Vec3
    size: Vec3
    rotation: Quaternion
    score: float
    prev: str
    next: str
    num_lidar_pts: int
    num_radar_pts: int
    attribute_tokens: StrTuple

    @classmethod
    def from_dict(cls, data: dict) -> SampleAnnotation:
        """Construct an instance from a dictionary.

        Args:
            data: A dictionary containing the record fields.

        Returns:
            An instance of the subclass.

        """
        return cls(
            translation=vec3(data.pop("translation")),
            size=vec3(data.pop("size")),
            rotation=quat(data.pop("rotation")),
            attribute_tokens=str_tuple(data.pop("attribute_tokens")),
            **data,
        )


@dataclass(slots=True)
class SampleAnnotation2D(Record):
    """A 2D bounding box annotation of an object in a camera sample_data.

    Attributes:
        token: Unique identifier of the record.
        sample_token: Token referencing the associated sample record.
        calibrated_sensor_token: Token referencing the associated calibrated sensor record.
            By this token we can infer which camera recorded this object.
            Sample data record with this calibrated_sensor_token and sample_token that is
            key frame gets us image on which this annotation was created.
        instance_token: Token referencing the associated 3D instance.
            If instance does not have 3D annotations instance record will not be present.
        instance_2d_token: Unique identifier for the 2D instance.
        corners: 2D bounding box represented as (min_x, min_y, max_x, max_y).
        score: Score of 2D annotation.
        prev: Token of the previous 2D annotation ("" if none).
        next: Token of the next 2D annotation ("" if none).
    """

    token: str
    sample_token: str
    calibrated_sensor_token: str
    instance_token: str
    instance_2d_token: str
    corners: Corners
    score: float
    prev: str
    next: str
    attribute_tokens: StrTuple

    @classmethod
    def from_dict(cls, data: dict) -> SampleAnnotation2D:
        """Construct an instance from a dictionary.

        Args:
            data: A dictionary containing the record fields.

        Returns:
            An instance of the subclass.

        """
        return cls(
            corners=corners(data.pop("corners")),
            attribute_tokens=str_tuple(data.pop("attribute_tokens")),
            **data,
        )


@dataclass(slots=True)
class SampleData(Record, Timestamp):
    """A sensor data record associated with a sample.

    Represents a single file captured by a sensor (LiDAR, camera, radar) in a sample.

    Attributes:
        token: Unique identifier of the record.
        sample_token: Token referencing the associated sample record.
        ego_pose_token: Token referencing the associated ego pose record.
        calibrated_sensor_token: Token referencing the associated calibrated sensor record.
        filename: Relative path from dataset root directory to the sensor data file.
        fileformat: File format (e.g., 'pcd', 'png').
        width: Width of the sensor image (0 for non-image sensors).
        height: Height of the sensor image (0 for non-image sensors).
        timestamp: Unix timestamp in microseconds.
        prev: Token of the previous sample data record ("" if none).
        next: Token of the next sample data record ("" if none).
        is_key_frame: Whether this frame is a keyframe.
        checksum: File checksum for integrity verification.
        channel: Sensor channel name (e.g., "RADAR_FRONT_LEFT").
        modality: Sensor modality (e.g., "radar", "lidar", "camera").
        anns: List of 2D annotations present in this sample data record (only camera channels will have some)

    """

    token: str
    sample_token: str
    ego_pose_token: str
    calibrated_sensor_token: str
    filename: str
    fileformat: str
    width: int
    height: int
    timestamp: int
    prev: str
    next: str
    is_key_frame: bool
    checksum: str
    channel: str = ""
    modality: str = ""
    anns: StrTuple = field(default_factory=tuple)


@dataclass(slots=True)
class Scene(Record):
    """A scene in the dataset, representing a continuous sequence of samples.

    Attributes:
        token: Unique identifier of the record.
        name: Human-readable name of the scene.
        description: Optional description of the scene.
        nbr_samples: Number of samples in the scene.
        first_sample_token: Token of the first sample in the scene.
        last_sample_token: Token of the last sample in the scene.
        scene_metadata_token: Token referencing the associated scene metadata record.
    """

    token: str
    name: str
    description: str
    nbr_samples: int
    first_sample_token: str
    last_sample_token: str
    scene_metadata_token: str = ""


@dataclass(slots=True)
class Attribute(Record):
    """Attribute of the dataset.

    Attributes:
        token: Unique identifier of the record.
        name: Human-readable name of the attribute.
        description: Optional description of the attribute.
    """

    token: str
    name: str
    description: str


@dataclass(slots=True)
class Sensor(Record):
    """A sensor in the dataset.

    Represents a physical sensor mounted on the ego vehicle.

    Attributes:
        token: Unique identifier of the record.
        channel: Sensor channel name (e.g., "RADAR_FRONT_LEFT").
        modality: Sensor type/modality (e.g., "radar", "lidar", "camera").
    """

    token: str
    channel: str
    modality: str
