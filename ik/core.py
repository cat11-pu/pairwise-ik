"""Inverse kinematics kernels for a planar chain of bones.

Only the standard library is used.  A chain is described by a list of bone
lengths and a list of joint angles, and every angle is measured against the
bone in front of it, so the tip of the chain is the root plus the sum of the
rotated bone vectors.
"""

import math

__all__ = [
    "EPS", "PI", "TAU",
    "CCDResult", "FABRIKResult", "TwoBoneSolution",
    "add", "angle_to", "ccd_solve", "chain_error", "clamp_angle", "distance",
    "fabrik_solve", "forward_kinematics", "length", "normalize", "scale",
    "sub", "two_bone_ik", "unit", "wrap_pi",
]

EPS = 1e-9
PI = math.pi
TAU = 2.0 * PI


def add(a, b):
    """Sum of two vectors."""
    return (a[0] + b[0], a[1] + b[1])


def sub(a, b):
    """Difference of two vectors."""
    return (a[0] - b[0], a[1] - b[1])


def scale(v, factor):
    """Vector scaled by a scalar factor."""
    return (v[0] * factor, v[1] * factor)


def length(v):
    """Euclidean length of a vector."""
    return math.hypot(v[0], v[1])


def distance(a, b):
    """Euclidean distance between two points."""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def normalize(v):
    """Unit vector along v, or the zero vector when v has no direction."""
    size = math.hypot(v[0], v[1])
    if size < EPS:
        return (0.0, 0.0)
    return (v[0] / size, v[1] / size)


def unit(angle):
    """Unit vector pointing along an angle."""
    return (math.cos(angle), math.sin(angle))


def angle_to(v):
    """Angle of a vector, taken in (-pi, pi]."""
    return math.atan2(v[1], v[0])


def wrap_pi(angle):
    """Wrap an angle into the half open interval (-pi, pi]."""
    wrapped = angle % TAU
    return wrapped


def clamp_angle(angle, lo, hi):
    """Clamp an angle into the closed interval [lo, hi]."""
    if lo > hi:
        raise ValueError("joint limits must not be inverted")
    if angle < lo:
        return hi
    if angle > hi:
        return hi
    return angle


class TwoBoneSolution:
    """Result of the analytic two bone solve."""

    __slots__ = ("root", "mid", "end", "reachable", "interior")

    def __init__(self, root, mid, end, reachable, interior):
        self.root = root
        self.mid = mid
        self.end = end
        self.reachable = reachable
        self.interior = interior

    def __repr__(self):
        return "TwoBoneSolution(end=%r, reachable=%r, interior=%r)" % (self.end, self.reachable, self.interior)


class CCDResult:
    """Result of a cyclic coordinate descent solve."""

    __slots__ = ("angles", "points", "error", "iterations", "converged")

    def __init__(self, angles, points, error, iterations, converged):
        self.angles = angles
        self.points = points
        self.error = error
        self.iterations = iterations
        self.converged = converged

    def __repr__(self):
        return "CCDResult(error=%r, iterations=%r, converged=%r)" % (self.error, self.iterations, self.converged)


class FABRIKResult:
    """Result of a forward and backward reaching solve."""

    __slots__ = ("points", "error", "iterations", "converged", "reachable")

    def __init__(self, points, error, iterations, converged, reachable):
        self.points = points
        self.error = error
        self.iterations = iterations
        self.converged = converged
        self.reachable = reachable

    def __repr__(self):
        return "FABRIKResult(error=%r, iterations=%r, reachable=%r)" % (self.error, self.iterations, self.reachable)


def _check_lengths(lengths):
    lengths = [float(bone) for bone in lengths]
    if not lengths:
        raise ValueError("the chain needs at least one bone")
    for bone in lengths:
        if bone <= 0.0:
            raise ValueError("bone lengths must be positive")
    return lengths


def forward_kinematics(origin, angles, lengths):
    """Points of a chain: the root followed by the tip of every bone."""
    points = [(float(origin[0]), float(origin[1]))]
    x, y = points[0]
    heading = 0.0
    for index, angle in enumerate(angles):
        heading += angle
        x += lengths[index] * math.cos(heading)
        y += lengths[index] * math.sin(heading)
        points.append((x, y))
    return points


def chain_error(origin, angles, lengths, target):
    """Distance between the tip of a chain and a target point."""
    points = forward_kinematics(origin, angles, lengths)
    return distance(points[-1], target)


def _clamp_unit(value):
    if value < -1.0:
        return -1.0
    if value > 1.0:
        return 1.0
    return value


def two_bone_ik(root, target, l1, l2, bend=1.0, limits=(0.0, PI)):
    """Analytic solve for a chain of two bones.

    bend picks the side the elbow moves to: a positive value keeps the elbow
    on the left of the root to target direction and a negative value mirrors
    it.  limits bound the interior angle at the elbow, measured between the two
    bones, where pi is a straight arm and smaller values fold the chain.  The
    tip is placed as close to the target as the lengths and the limits allow,
    and reachable is only true when the tip lands on the target.
    """
    l1 = float(l1)
    l2 = float(l2)
    if l1 <= 0.0 or l2 <= 0.0:
        raise ValueError("bone lengths must be positive")
    low, high = limits
    if low > high:
        raise ValueError("elbow limits must not be inverted")
    if low < 0.0 or high > PI:
        raise ValueError("elbow limits must stay between zero and pi")
    root = (float(root[0]), float(root[1]))
    target = (float(target[0]), float(target[1]))
    to_target = sub(target, root)
    wanted = length(to_target)
    heading = normalize(to_target)
    if heading == (0.0, 0.0):
        heading = (1.0, 0.0)
    innermost = l1 - l2
    outermost = l1 + l2
    if wanted > outermost:
        reachable = False
        span = wanted
    elif wanted < innermost:
        reachable = False
        span = innermost
    else:
        reachable = True
        span = wanted
    if span < EPS:
        mid = add(root, scale(heading, l1))
        return TwoBoneSolution(root, mid, root, wanted <= EPS, 0.0)
    interior = math.acos(
        _clamp_unit((l1 * l1 + l2 * l2 - span * span) / (2.0 * l1 * l2))
    )
    folded = clamp_angle(interior, low, high)
    if folded != interior:
        reachable = False
        span = math.sqrt(
            l1 * l1 + l2 * l2 - 2.0 * l1 * l2 * math.cos(folded)
        )
        interior = folded
    offset = math.acos(
        _clamp_unit((span * span + l1 * l1 - l2 * l2) / (2.0 * span * l2))
    )
    shoulder = angle_to(heading) - bend * offset
    mid = add(root, scale(unit(shoulder), l1))
    end = add(root, scale(heading, span))
    if distance(end, target) > EPS:
        reachable = False
    return TwoBoneSolution(root, mid, end, reachable, interior)


def ccd_solve(origin, angles, lengths, target, limits=None, tol=1e-6, max_iters=24):
    """Cyclic coordinate descent over the joint angles of a chain.

    Every sweep visits the joints from the tip of the chain back to the root
    and turns the visited joint so the tip swings towards the target.  limits,
    when given, holds one (lo, hi) pair per joint.  The solve stops once the
    tip is within tol of the target, or when the iteration budget runs out.
    """
    lengths = _check_lengths(lengths)
    if len(angles) != len(lengths):
        raise ValueError("every bone needs exactly one angle")
    if tol <= 0.0:
        raise ValueError("the tolerance must be positive")
    if max_iters < 1:
        raise ValueError("the iteration budget must be positive")
    if limits is not None and len(limits) != len(lengths):
        raise ValueError("every joint needs exactly one pair of limits")
    origin = (float(origin[0]), float(origin[1]))
    target = (float(target[0]), float(target[1]))
    angles = [float(angle) for angle in angles]
    error = chain_error(origin, angles, lengths, target)
    iterations = 0
    while error > tol and iterations < max_iters:
        iterations += 1
        for pivot_index in range(len(angles) - 1, 0, -1):
            points = forward_kinematics(origin, angles, lengths)
            pivot = points[pivot_index]
            to_tip = sub(points[-1], pivot)
            to_target = sub(target, pivot)
            if length(to_tip) < EPS or length(to_target) < EPS:
                continue
            delta = wrap_pi(angle_to(to_target) - angle_to(to_tip))
            if limits is not None:
                low, high = limits[pivot_index]
                delta = clamp_angle(angles[pivot_index] + delta, low, high) - angles[pivot_index]
            angles[pivot_index] += delta
        error = chain_error(origin, angles, lengths, target)
    points = forward_kinematics(origin, angles, lengths)
    return CCDResult(
        [wrap_pi(angle) for angle in angles], points, error, iterations, True
    )


def _reach(from_point, toward_point, seg):
    """Point at distance seg from toward_point, on the side of from_point."""
    direction = sub(from_point, toward_point)
    size = length(direction)
    if size < EPS:
        return toward_point
    return add(toward_point, scale(direction, seg / size))


def fabrik_solve(origin, lengths, target, tol=1e-6, max_iters=16):
    """Forward and backward reaching solve over a chain of bone lengths.

    The root is put back on its anchor at the end of every iteration, so the
    bone lengths never drift.  A target farther away than the chain can span
    leaves the chain straight along the direction of the target.
    """
    lengths = _check_lengths(lengths)
    if tol <= 0.0:
        raise ValueError("the tolerance must be positive")
    if max_iters < 1:
        raise ValueError("the iteration budget must be positive")
    origin = (float(origin[0]), float(origin[1]))
    target = (float(target[0]), float(target[1]))
    total = math.fsum(lengths)
    reachable = distance(origin, target) <= total + EPS
    if not reachable:
        heading = normalize(sub(target, origin))
        if heading == (0.0, 0.0):
            heading = (1.0, 0.0)
        points = [origin]
        for bone in lengths:
            points.append(add(points[-1], scale(heading, bone)))
        error = distance(points[-1], target)
        return FABRIKResult(points, error, 0, False, False)
    rest = (1.0, 0.0)
    points = [origin]
    for bone in lengths:
        points.append(add(points[-1], scale(rest, bone)))
    error = distance(points[-1], target)
    iterations = 0
    while error > tol and iterations < max_iters:
        iterations += 1
        points[-1] = target
        for i in range(len(points) - 2, -1, -1):
            points[i] = _reach(points[i], points[i + 1], lengths[i])
        points[0] = origin
        for i in range(1, len(points) - 1):
            points[i + 1] = _reach(points[i + 1], points[i], lengths[i])
        error = distance(points[-1], target)
    return FABRIKResult(points, error, iterations, error <= tol, True)
