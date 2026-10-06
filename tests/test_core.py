"""Behaviour tests for the inverse kinematics kernels."""

import math
import unittest

from ik.core import (
    ccd_solve,
    clamp_angle,
    distance,
    fabrik_solve,
    forward_kinematics,
    two_bone_ik,
    wrap_pi,
)


class AngleTests(unittest.TestCase):
    def test_angles_are_wrapped_into_the_left_half_plane(self):
        self.assertAlmostEqual(wrap_pi(0.25), 0.25, places=12)
        self.assertAlmostEqual(wrap_pi(-0.5), -0.5, places=12)
        self.assertAlmostEqual(wrap_pi(-math.pi / 2.0), -math.pi / 2.0, places=12)
        self.assertAlmostEqual(wrap_pi(1.5 * math.pi), -0.5 * math.pi, places=12)
        self.assertAlmostEqual(wrap_pi(math.pi), math.pi, places=12)
        self.assertAlmostEqual(wrap_pi(-math.pi), math.pi, places=12)


class TwoBoneTests(unittest.TestCase):
    def test_elbow_limits_bend_the_chain_without_stretching_it(self):
        limit = 1.0
        solution = two_bone_ik((0.0, 0.0), (0.25, 0.0), 1.0, 1.0, limits=(limit, math.pi))
        self.assertAlmostEqual(solution.interior, limit, places=9)
        span = math.sqrt(
            1.0 * 1.0 + 1.0 * 1.0 - 2.0 * 1.0 * 1.0 * math.cos(limit)
        )
        self.assertAlmostEqual(distance(solution.root, solution.end), span, places=9)
        self.assertAlmostEqual(solution.end[0], span, places=9)
        self.assertAlmostEqual(solution.end[1], 0.0, places=9)
        self.assertFalse(solution.reachable)
        offset = math.acos(
            min(max((span * span + 1.0 * 1.0 - 1.0 * 1.0) / (2.0 * span * 1.0), -1.0), 1.0)
        )
        self.assertAlmostEqual(distance(solution.root, solution.mid), 1.0, places=9)
        self.assertAlmostEqual(solution.mid[0], math.cos(offset), places=9)
        self.assertAlmostEqual(solution.mid[1], math.sin(offset), places=9)

    def test_two_bone_elbow_preserves_both_bone_lengths(self):
        solution = two_bone_ik((0.0, 0.0), (2.0, 0.0), 2.0, 1.0)
        self.assertTrue(solution.reachable)
        self.assertAlmostEqual(distance(solution.root, solution.mid), 2.0, places=9)
        self.assertAlmostEqual(distance(solution.mid, solution.end), 1.0, places=9)
        self.assertAlmostEqual(solution.end[0], 2.0, places=9)
        self.assertAlmostEqual(solution.end[1], 0.0, places=9)
        interior = math.acos((2.0 * 2.0 + 1.0 * 1.0 - 2.0 * 2.0) / (2.0 * 2.0 * 1.0))
        self.assertAlmostEqual(solution.interior, interior, places=9)
        cross = (solution.end[0] - solution.root[0]) * (solution.mid[1] - solution.root[1]) - (
            solution.end[1] - solution.root[1]
        ) * (solution.mid[0] - solution.root[0])
        self.assertGreater(cross, 0.0)

    def test_bend_sign_chooses_the_side_of_the_elbow(self):
        solution = two_bone_ik((0.0, 0.0), (1.0, 1.0), 1.0, 1.0, bend=1.0)
        self.assertAlmostEqual(solution.mid[0], 0.0, places=9)
        self.assertAlmostEqual(solution.mid[1], 1.0, places=9)
        cross = (solution.end[0] - solution.root[0]) * (solution.mid[1] - solution.root[1]) - (
            solution.end[1] - solution.root[1]
        ) * (solution.mid[0] - solution.root[0])
        self.assertGreater(cross, 0.0)
        mirrored = two_bone_ik((0.0, 0.0), (1.0, 1.0), 1.0, 1.0, bend=-1.0)
        self.assertAlmostEqual(mirrored.mid[0], 1.0, places=9)
        self.assertAlmostEqual(mirrored.mid[1], 0.0, places=9)

    def test_target_beyond_the_arm_leaves_it_stretched(self):
        solution = two_bone_ik((0.0, 0.0), (5.0, 0.0), 1.25, 1.25)
        self.assertFalse(solution.reachable)
        self.assertAlmostEqual(distance(solution.root, solution.end), 2.5, places=9)
        self.assertAlmostEqual(distance(solution.root, solution.mid), 1.25, places=9)
        self.assertAlmostEqual(distance(solution.mid, solution.end), 1.25, places=9)
        self.assertAlmostEqual(solution.end[0], 2.5, places=9)
        self.assertAlmostEqual(solution.end[1], 0.0, places=9)
        self.assertGreater(distance(solution.end, (5.0, 0.0)), 2.0)

    def test_target_inside_the_length_gap_cannot_be_reached(self):
        solution = two_bone_ik((0.0, 0.0), (1.0, 0.0), 1.0, 3.0)
        self.assertFalse(solution.reachable)
        self.assertAlmostEqual(solution.interior, 0.0, places=9)
        self.assertAlmostEqual(distance(solution.root, solution.mid), 1.0, places=9)
        self.assertAlmostEqual(distance(solution.root, solution.end), 2.0, places=9)
        self.assertAlmostEqual(solution.end[0], 2.0, places=9)
        self.assertAlmostEqual(solution.end[1], 0.0, places=9)


class CCDTests(unittest.TestCase):
    def test_ccd_turns_the_tip_onto_a_target_behind_the_root(self):
        origin = (0.0, 0.0)
        lengths = [1.0, 1.0, 1.0]
        target = (-1.5, 1.5)
        result = ccd_solve(origin, [0.0, 0.0, 0.0], lengths, target, tol=1e-9, max_iters=64)
        self.assertTrue(result.converged)
        self.assertLessEqual(result.error, 1e-9)
        self.assertAlmostEqual(distance(result.points[-1], target), result.error, places=12)
        for angle in result.angles:
            self.assertGreaterEqual(angle, -math.pi - 1e-12)
            self.assertLessEqual(angle, math.pi + 1e-12)
        again = forward_kinematics(origin, result.angles, lengths)
        self.assertAlmostEqual(distance(again[-1], result.points[-1]), 0.0, places=9)

    def test_ccd_reports_a_spent_iteration_budget(self):
        result = ccd_solve(
            (0.0, 0.0),
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            (0.25, 2.0),
            tol=1e-9,
            max_iters=2,
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 2)
        self.assertGreater(result.error, 1e-9)


class FABRIKTests(unittest.TestCase):
    def test_fabrik_reaches_inside_its_span_and_keeps_bone_lengths(self):
        origin = (0.0, 0.0)
        lengths = [1.0, 1.0, 1.0]
        result = fabrik_solve(origin, lengths, (1.5, 1.0), tol=1e-9, max_iters=32)
        self.assertTrue(result.reachable)
        self.assertTrue(result.converged)
        self.assertLessEqual(result.error, 1e-9)
        self.assertEqual(result.points[0], origin)
        for index, bone in enumerate(lengths):
            self.assertAlmostEqual(
                distance(result.points[index], result.points[index + 1]), bone, places=9
            )
        far = fabrik_solve(origin, lengths, (10.0, 0.0), tol=1e-9, max_iters=32)
        self.assertFalse(far.reachable)
        self.assertFalse(far.converged)
        self.assertAlmostEqual(far.error, 7.0, places=9)
        self.assertAlmostEqual(far.points[-1][0], 3.0, places=9)
        self.assertAlmostEqual(far.points[-1][1], 0.0, places=9)
        for index, bone in enumerate(lengths):
            self.assertAlmostEqual(
                distance(far.points[index], far.points[index + 1]), bone, places=9
            )


class ValidationTests(unittest.TestCase):
    def test_invalid_chain_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            two_bone_ik((0.0, 0.0), (1.0, 0.0), 0.0, 1.0)
        with self.assertRaises(ValueError):
            two_bone_ik((0.0, 0.0), (1.0, 0.0), 1.0, 1.0, limits=(2.0, 0.5))
        with self.assertRaises(ValueError):
            clamp_angle(0.5, 1.0, 0.0)
        with self.assertRaises(ValueError):
            ccd_solve((0.0, 0.0), [0.0, 0.0], [1.0], (1.0, 1.0))
        with self.assertRaises(ValueError):
            ccd_solve((0.0, 0.0), [0.0], [1.0], (1.0, 1.0), tol=0.0)
        with self.assertRaises(ValueError):
            fabrik_solve((0.0, 0.0), [1.0, -1.0], (1.0, 1.0))
        with self.assertRaises(ValueError):
            fabrik_solve((0.0, 0.0), [1.0], (1.0, 1.0), max_iters=0)


if __name__ == "__main__":
    unittest.main()
