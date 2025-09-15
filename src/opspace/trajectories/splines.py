"""Splines / "composite Bezier curves" composed of multiple chained Bezier curves

Initial boundary conditions are imposed on the start of the first curve, final B.C.s are imposed at the end of
the third curve, and we enforce C2 derivative continuity at the knot points

Reference:
Fast Path Planning Through Large Collections of Safe Boxes
https://github.com/cvxgrp/fastpathplanning/blob/main/fastpathplanning/smooth.py
https://web.stanford.edu/~boyd/papers/pdf/fpp.pdf
"""

from typing import Union

import cvxpy as cp
import numpy as np

from src.opspace.trajectories.bezier import BezierCurve


class CompositeBezierCurve:
    """Composite Bezier curve class for a continuous chain of connected Bezier curves

    Args:
        beziers (list[BezierCurve]): Consecutive Bezier curves composing the composite curve.
    """

    def __init__(self, beziers: list[BezierCurve]):
        for bez1, bez2 in zip(beziers[:-1], beziers[1:]):
            assert bez1.b == bez2.a
            assert bez1.d == bez2.d

        self.beziers = beziers
        self.N = len(self.beziers)
        self.d = beziers[0].d
        self.a = beziers[0].a
        self.b = beziers[-1].b
        self.duration = self.b - self.a
        self.transition_times = np.array([self.a] + [bez.b for bez in beziers])
        self.segment_durations = np.array([bez.duration for bez in beziers])

    def find_segment(self, t):
        # return min(bisect(self.transition_times, t) - 1, self.N - 1)
        # TODO check to see if this will work on a cp variable (probably not...)
        return np.minimum(
            np.searchsorted(self.transition_times, t, "right") - 1, self.N - 1
        )

    def __call__(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluates the composite Bezier curve at specified points, by calling each of the child curves

        Args:
            t (Union[float, npt.ArrayLike]): Evaluation points (for instance, trajectory times)

        Returns:
            np.ndarray: Points along the composite curve, shape (n_pts, dimension)
        """
        seg_map = self.find_segment(t)
        evals = []
        for i in range(self.N):
            evals.append(self.beziers[i](t[seg_map == i]))
        return np.row_stack(evals)

    @property
    def start_point(self):
        """Starting point of the composite Bezier curve (the first point of the first curve)"""
        return self.beziers[0].start_point

    @property
    def end_point(self):
        """Ending point of the composite Bezier curve (the last point of the last curve)"""
        return self.beziers[-1].end_point

    @property
    def derivative(self):
        """Derivative of the composite Bezier curve (a composite Bezier curve of derivative curves)"""
        return CompositeBezierCurve([b.derivative for b in self.beziers])

    @property
    def l2_squared_sum(self) -> Union[float, cp.Expression]:
        """Sum of the squared L2 norm for all curves"""
        return sum(bez.l2_squared for bez in self.beziers)
