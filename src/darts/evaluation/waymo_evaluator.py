"""DARTS evaluation based on The Waymo Open Dataset."""
# /* Copyright 2019 The Waymo Open Dataset Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================*/

# // Copyright 2011 Google Inc. All Rights Reserved.
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyquaternion import Quaternion

from darts.dataset.user_models import Box

from .registry import EvaluateInterface, register_evaluator

if TYPE_CHECKING:
    from darts.dataset.darts import DARTS
    from darts.dataset.dataset_models import SampleAnnotation
    from darts.dataset.user_models import DARTSAnnotations

K_CROSS_EXACT_EPSILON = 2**-50
K_EPSILON = 1e-10
K_MIN_BOX_DIM = 1e-2
logger = logging.getLogger(__name__)


@dataclass()
class Vec2D:
    """Vec2D class used mainly for 2D IOU computation."""

    x: float
    y: float

    def __add__(self, other: Vec2D) -> Vec2D:
        """Calculates addition product between two 2D vectors.

        Args:
            other:other vector
        Returns:
            Vec2D
        """
        return Vec2D(
            self.x + other.x,
            self.y + other.y,
        )

    def __sub__(self, other: Vec2D) -> Vec2D:
        """Calculates substraction between two 2D vectors.

        Args:
            other:other vector
        Returns:
            Vec2D
        """
        return Vec2D(self.x - other.x, self.y - other.y)

    def cross_prod(self, other: Vec2D) -> float:
        """Calculates cross product between two 2D vectors.

        Args:
            other:other vector
        Returns:
            cross product
        """
        return self.x * other.y - self.y * other.x


@dataclass()
class Polygon2D:
    """Polygon2D class used mainly for 2D IOU computation."""

    points: list[Vec2D]

    bbox_bottom_left: Vec2D = field(init=False)
    bbox_top_right: Vec2D = field(init=False)
    min_x: float = field(init=False)
    min_y: float = field(init=False)
    max_x: float = field(init=False)
    max_y: float = field(init=False)
    number_of_vertices: int = field(init=False)
    min_closed_shape: int = 3

    def __post_init__(self) -> None:
        """Cache polygon bounding box and vertex count."""
        self.min_x, self.min_y, self.max_x, self.max_y = self._compute_aabb()
        self.bbox_bottom_left = Vec2D(self.min_x, self.min_y)
        self.bbox_top_right = Vec2D(self.max_x, self.max_y)
        self.number_of_vertices = len(self.points)

    def _next(self, idx: int) -> int:
        return (idx + 1) % self.number_of_vertices

    def _prev(self, idx: int) -> int:
        return (idx - 1) % self.number_of_vertices

    @staticmethod
    def from_box(box: Box) -> Polygon2D:
        """Creates Polygon2D instance from Box instance.

        Args:
            box:box
        Returns:
            cross Polygon2D
        """
        cx, cy, _ = box.center
        length_x, lenght_y, _ = box.size

        q = Quaternion(w=box.orientation[0], x=box.orientation[1], y=box.orientation[2], z=box.orientation[3])

        yaw = q.yaw_pitch_roll[0]

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        local = [
            (length_x / 2, -lenght_y / 2),
            (length_x / 2, lenght_y / 2),
            (-length_x / 2, lenght_y / 2),
            (-length_x / 2, -lenght_y / 2),
        ]

        pts: list[Vec2D] = []
        for x, y in local:
            rx = cos_y * x - sin_y * y + cx
            ry = sin_y * x + cos_y * y + cy
            pts.append(Vec2D(rx, ry))

        return Polygon2D(pts)

    def _compute_aabb(self) -> tuple[float, float, float, float]:
        min_x = min(p.x for p in self.points)
        max_x = max(p.x for p in self.points)
        min_y = min(p.y for p in self.points)
        max_y = max(p.y for p in self.points)
        return min_x, min_y, max_x, max_y

    def _maybe_has_intersection_with(self, other: Polygon2D) -> bool:

        # separation axis theorem (AABB version)
        return not (
            self.min_x > other.max_x or other.min_x > self.max_x or self.min_y > other.max_y or other.min_y > self.max_y
        )

    def _cross3(self, v0: Vec2D, v1: Vec2D, v2: Vec2D) -> float:
        return (v1 - v0).cross_prod(v2 - v0)

    def _cross4(self, v0: Vec2D, v1: Vec2D, v2: Vec2D, v3: Vec2D) -> float:
        return (v1 - v0).cross_prod(v3 - v2)

    def _cross_maybe_exact(self, v0: Vec2D, v1: Vec2D, v2: Vec2D, v3: Vec2D) -> float:
        # degenerate edges
        if v0 == v1 or v2 == v3:
            return 0.0

        c = self._cross4(v0, v1, v2, v3)
        if abs(c) > K_CROSS_EXACT_EPSILON:
            return c

        term = [0.0] * 16
        # main products
        term[0] = v1.x * v3.y
        term[1] = -v0.x * v3.y
        term[2] = -v1.x * v2.y
        term[3] = v0.x * v2.y
        term[4] = -v1.y * v3.x
        term[5] = v0.y * v3.x
        term[6] = v1.y * v2.x
        term[7] = -v0.y * v2.x

        # correction terms using fma (if available)
        term[8] = math.fma(v1.x, v3.y, -term[0])
        term[9] = math.fma(-v0.x, v3.y, -term[1])
        term[10] = math.fma(-v1.x, v2.y, -term[2])
        term[11] = math.fma(v0.x, v2.y, -term[3])
        term[12] = math.fma(-v1.y, v3.x, -term[4])
        term[13] = math.fma(v0.y, v3.x, -term[5])
        term[14] = math.fma(v1.y, v2.x, -term[6])
        term[15] = math.fma(-v0.y, v2.x, -term[7])

        # precise summation (Kahan-style)
        total = 0.0
        comp = 0.0
        for t in term:
            y = t - comp
            temp = total + y
            comp = (temp - total) - y
            total = temp

        return total

    def _bound_to_range(self, lo: float, hi: float, x: float) -> float:
        return max(lo, min(hi, x))

    def _colinear_segment_point_intersection(self, p0: Vec2D, p1: Vec2D, q: Vec2D) -> tuple[bool, Vec2D | None]:
        # Element-wise min/max
        min_x = min(p0.x, p1.x)
        min_y = min(p0.y, p1.y)
        max_x = max(p0.x, p1.x)
        max_y = max(p0.y, p1.y)

        if (min_x <= q.x <= max_x) and (min_y <= q.y <= max_y):
            return True, q  # exact point, no modification

        return False, None

    # Compute the intersection of two segments. The algorithm in
    # compute_intersection_vertices depends on exact computation, adjustment by
    # epsilon is not acceptable.
    def _exact_intersection(self, p0: Vec2D, p1: Vec2D, q0: Vec2D, q1: Vec2D, det: float) -> tuple[bool, Vec2D | None]:  # noqa: PLR0911, C901
        if det == 0.0:
            # Segments are parallel.
            if self._cross_maybe_exact(p0, p1, p0, q0) != 0.0:
                # But not colinear.
                return False, None

            # If they are overlapping, return one of the intersection points.
            for a, b, c in [(q0, q1, p0), (q0, q1, p1), (p0, p1, q0), (p0, p1, q1)]:
                ok, pt = self._colinear_segment_point_intersection(a, b, c)
                if ok:
                    return True, pt

            return False, None

        detsign = math.copysign(1.0, det)
        epsilon = max(abs(det), 1.0) * K_CROSS_EXACT_EPSILON

        # t1/det is the intersection point projected to s1.
        # Must be in the range [0-1] if there is an intersection
        t1 = self._cross_maybe_exact(p0, q0, q1, q0)
        if t1 * detsign < 0.0 or t1 * detsign > abs(det) + epsilon:
            return False, None

        if t1 * detsign > abs(det) - epsilon:
            # t1/det is close to 1, reverse both segments and recompute the
            # intersection, to use full precision if close to the segment start.
            rt1 = self._cross_maybe_exact(p1, q1, q0, q1)
            if rt1 * detsign < 0.0:
                return False, None

        # t2/det is the intersection point projected to s2.
        # Must be in the range [0-1] if there is an intersection
        t2 = self._cross_maybe_exact(p0, p1, p0, q0)
        if t2 * detsign < 0.0 or t2 * detsign > abs(det) + epsilon:
            return False, None

        if t2 * detsign > abs(det) - epsilon:
            # t2/det is close to 1, reverse both segments and recompute the
            # intersection, to use full precision if close to the segment start.
            rt2 = self._cross_maybe_exact(p1, p0, p1, q1)
            if rt2 * detsign < 0.0:
                return False, None

        # bound_to_range to avoid precision errors from the division.
        t = self._bound_to_range(0.0, 1.0, t1 / det)

        ix = p0.x + (p1.x - p0.x) * t
        iy = p0.y + (p1.y - p0.y) * t

        return True, Vec2D(ix, iy)

    def _segment(self, i: int) -> Vec2D:
        return self.points[self._next(i)] - self.points[i]

    def _point_inside(self, p: Vec2D) -> bool:
        if self.number_of_vertices < self.min_closed_shape:
            return False
        if (
            p.x < self.bbox_bottom_left.x
            or p.y < self.bbox_bottom_left.y
            or p.x > self.bbox_top_right.x
            or p.y > self.bbox_top_right.y
        ):
            return False
        start_ix = 0
        end_ix = self.number_of_vertices
        # For small polygons / ranges, e.g. polygon with 10 vertices, the constants
        # of the O(log(n)) algorithm are sufficienly high that the O(n) algorithm
        # is faster.
        k_linear_search_threshold = 10
        # Divide the polygon into two and decide which side we should continue
        # checking. As invariant, the polygon we consider has vertices
        # points_[0], points_[start_ix], ... points_[end_ix].
        while end_ix - start_ix > k_linear_search_threshold:
            mid_ix = (end_ix - start_ix) % 2 + start_ix
            bisector: Vec2D = self.points[mid_ix] - self.points[0]
            if bisector.cross_prod(p - self.points[0]) >= 0:
                start_ix = mid_ix
            else:
                end_ix = mid_ix

        for ii in range(start_ix, end_ix - 1):
            s: Vec2D = self.points[ii + 1] - self.points[ii]
            if s.cross_prod(p - self.points[ii]) < 0.0:
                return False

        return not (start_ix < end_ix and self._segment(end_ix - 1).cross_prod(p - self.points[end_ix - 1]) < 0.0)

    def _compute_intersection_vertices(self, other: Polygon2D) -> list[Vec2D] | None:  # noqa: PLR0915, PLR0912, C901
        if not self._maybe_has_intersection_with(other):
            return None
        # This algorithm is from the following paper:
        #
        # O'Rourke, Joseph, et al. "A new linear algorithm for intersecting convex
        # polygons." Computer Graphics and Image Processing 19.4 (1982): 384-391.
        #
        # The algorithm can be viewed as a geometric generalization of merging two
        # sorted lists. It performs a counter-clockwise traversal of the boundaries
        # of the two polygons. The algorithm maintains a pair of edges, one from each
        # polygon. From a consideration of the relative positions of these edges the
        # algorithm advances one of them to the next edge in counterclockwise order
        # around its polygon. Intuitively, this is done in such a way that these two
        # edges effectively “chase” each other around the boundary of the
        # intersection polygon.

        # We use P and Q to represent this polygon and the other one.
        p_inside, q_inside = False, False
        total_number_of_points = self.number_of_vertices + other.number_of_vertices
        convex_points: list[Vec2D] = []

        # A lambda that is used to append a new convex point to the list. We ignore
        # the point that is identical to either the first or the last convex point in
        # the list to handle corner cases, in which two polygons may "intersect" with
        # either a point or an edge.
        def append_convex_point(p: Vec2D, convex_pts: list[Vec2D]) -> None:
            if len(convex_pts) == 0 or (p != convex_pts[0] and p != convex_pts[-1]):
                convex_pts.append(p)

        p_idx, q_idx = 0, 0
        p_idx_next, q_idx_next = 1, 1
        p0 = self.points[p_idx]
        p1 = self.points[p_idx_next]
        q0 = other.points[q_idx]
        q1 = other.points[q_idx_next]
        i = 0
        max_i = total_number_of_points * 2

        while i < max_i:
            # If we found a intersection between <p0, p1> and <q0, q1>, put the
            # intersection into the convex points list, and check that after the
            # intersection whose points will be inside and hence recorded. If we reach
            # the first intersection point, we are done.
            det = self._cross_maybe_exact(p0, p1, q1, q0)
            found, inter = self._exact_intersection(p0, p1, q0, q1, det)
            if found and inter is not None:
                if len(convex_points) >= self.min_closed_shape and inter == convex_points[0]:
                    return convex_points
                if len(convex_points) == 0:
                    # When we found the first intersection, we only need to iterate this
                    # loop at most total_num_points + 1 times to get back to this
                    # intersection.
                    i = total_number_of_points - 1
                append_convex_point(inter, convex_points)
                # If p1 is on the left of <q0, q1>, mark P as inside; otherwise mark Q as
                # inside.
                if self._cross_maybe_exact(q1, q0, p1, p0) >= 0:
                    p_inside = True
                    q_inside = False
                else:
                    p_inside = False
                    q_inside = True
            # Determine in which polygon we would like to advance according to the
            # "advance rule" in the algorithm.
            if det >= 0:
                advance_p = self._cross_maybe_exact(q1, q0, p1, p0) < 0
            else:
                advance_p = self._cross_maybe_exact(p1, p0, q1, q0) >= 0
            if advance_p:
                if p_inside:
                    append_convex_point(p1, convex_points)
                p_idx = p_idx_next
                p0 = p1
                p_idx_next = self._next(p_idx_next)
                p1 = self.points[p_idx_next]
            else:
                if q_inside:
                    append_convex_point(q1, convex_points)
                q_idx = q_idx_next
                q0 = q1
                q_idx_next = other._next(q_idx_next)
                q1 = other.points[q_idx_next]
            i += 1
        # Handle the case that those two polygons share one vertex or (part of) one
        # edge, in which case the intersection is empty.
        if len(convex_points) > 0:
            return []
        # Now we have three cases:
        #   1. P is inside of Q.
        #   2. Q is inside of P.
        #   3. P and Q don't intersect.

        # Check the case 1.
        p_is_inside = True
        for point in self.points:
            if not other._point_inside(point):
                p_is_inside = False
                break
        if p_is_inside:
            return self.points
        # Check the case 2.
        q_is_inside = True
        for point in other.points:
            if not self._point_inside(point):
                q_is_inside = False
                break
        if q_is_inside:
            return other.points
        # P and Q don't intersect.
        return None

    # Compute the area of a non self-instersecting polygon, which is represented by
    # all its vectices in counter-clockwise order.
    def _area_internal(self, points: list[Vec2D]) -> float:
        # Convex polygons must be simple polygons (i.e. without self-intersections).
        # One idea to compute the area of a simple polygon with N sides is to sum the
        # areas of N triangles where each triangle is formed by one polygon side and
        # an arbitrary 2D point. After shifting the polygon to one of its points, if
        # all the points lie in counter-clock direction all the triangle areas should
        # be positive. We choose this arbitrary point to be the origin points[0] so
        # that the area of each triangle could be easily computed from the cross
        # product of two vectors. Please see this page for more detailed explanation.
        # http://en.wikipedia.org/wiki/Polygon
        area = 0.0
        # Shift points to x_0, y_0 for precision in calculating area.
        # Starting from i = 1 because cross_prod for (x_0, y_0) is zero.
        num_points = len(points)
        for i in range(1, num_points - 1):
            j = i + 1
            area += self._cross3(points[0], points[i], points[j])
        if abs(area) <= K_EPSILON:
            area = 0
        return area * 0.5

    def compute_intersection_area(self, other: Polygon2D) -> float:
        """Computes intersection area between two polygons.

        Args:
            other: polygon to count intersection area against
        Returns:
            Intersection area
        """
        convex_points = self._compute_intersection_vertices(other)
        if convex_points is None:
            return 0.0
        return self._area_internal(convex_points)


@register_evaluator("WaymoEvaluator")
class WaymoEvaluator(EvaluateInterface):
    """WaymoEvaluator class."""

    def evaluate(self, darts: DARTS, annotations: DARTSAnnotations) -> str:
        """Evaluate annotations with WaymoEvaluator.

        :param darts: DARTS database
        :param annotations: Annotations created by user
        :return: evaluation results
        """
        gt = darts.sample_annotation.all()[0]
        for sample_list in annotations.sequences.values():
            for sample in sample_list:
                for box in sample.boxes:
                    logger.info(self._compute_iou(self._sample_annotations_to_box(darts, gt), box))
        return "END"

    def _sample_annotations_to_box(self, darts: DARTS, sample_annotation: SampleAnnotation) -> Box:
        category = darts.get_category_from_annotation(sample_annotation.token)
        return Box(
            center=list(sample_annotation.translation),
            size=list(sample_annotation.size),
            orientation=list(sample_annotation.rotation),
            name=category.name,
            score=1,
            track_id=1,
        )

    def _closed_ranges_overlap(
        self,
        min1: float,
        max1: float,
        min2: float,
        max2: float,
    ) -> tuple[bool, float, float]:

        overlap_min = max(min1, min2)
        overlap_max = min(max1, max2)

        overlap_exists = overlap_min <= overlap_max

        if overlap_exists:
            return True, overlap_min, overlap_max
        return False, overlap_min, overlap_max

    def _compute_iou(self, b1: Box, b2: Box) -> float:
        for size in b1.size:
            if size < K_MIN_BOX_DIM:
                return 0.0
        for size in b2.size:
            if size < K_MIN_BOX_DIM:
                return 0.0

        is_close_overlap, z_overlap_min, z_overlap_max = self._closed_ranges_overlap(
            b1.center[2] - b1.size[2] * 0.5,
            b1.center[2] + b1.size[2] * 0.5,
            b2.center[2] - b2.size[2] * 0.5,
            b2.center[2] + b2.size[2] * 0.5,
        )
        if not is_close_overlap:
            return 0.0
        p1 = Polygon2D.from_box(b1)
        p2 = Polygon2D.from_box(b2)
        intersection_area = p1.compute_intersection_area(p2)
        intersection_volume = intersection_area * (z_overlap_max - z_overlap_min)
        b1_volume = b1.size[0] * b1.size[1] * b1.size[2]
        b2_volume = b2.size[0] * b2.size[1] * b2.size[2]
        union_volume = b1_volume + b2_volume - intersection_volume
        if union_volume < K_EPSILON:
            return 0.0
        iou = intersection_volume / union_volume
        return max(min(iou, 1), 0)
