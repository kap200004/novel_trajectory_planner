
from mriToLines import run
import numpy as np
import timeit
import cmath
from functools import cmp_to_key
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Wedge
from matplotlib.collections import LineCollection
from convex_hull_tree import build_tree, get_hulls
from sys import getsizeof
from convex_hull_tree_FC import build_treefc


def get_size(obj, seen=None):
    """Recursively find the size of an object in bytes, including its referenced objects."""
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    seen.add(obj_id)
    size = getsizeof(obj)

    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(vars(obj), seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])

    return size


def get_total_size_of_tree(node, seen=None):
    """Recursively find the size of the entire tree in bytes, including all nodes and their contents."""
    if seen is None:
        seen = set()

    if node is None:
        return 0

    size = get_size(node, seen)

    if node.left_child:
        size += get_total_size_of_tree(node.left_child, seen)
    if node.right_child:
        size += get_total_size_of_tree(node.right_child, seen)

    return size
RADIUS = 30
line_segments = run('outline.tif', 117, 187)
target = (0, 0)
QUADRANTS = {(True, True):1, (False, True):2, (False, False):3, (True, False):4}


def calculate_envelopecc(ranges, s, e, is_ls):
    if s < e:
        middle = (s + e) // 2

        lenv = calculate_envelopecc(ranges, s, middle, is_ls)
        renv = calculate_envelopecc(ranges, middle + 1, e, is_ls)
        envelope = merge_envcc(lenv, renv, is_ls)
        return envelope
    if s > len(ranges)-1:
        return []
    else:
        return [ranges[s]]

def merge_envcc(lenv, renv, is_ls):

    envelope = []
    i, j = 0, 0

    while i < len(lenv) and j < len(renv):
        left_line, lRange = lenv[i]
        right_line, rRange = renv[j]
        lstart, lend = lRange
        rstart, rend = rRange

        # no overlap at all between the two ranges
        if not overlap(lRange, rRange):
            if lRange[0] < rRange[0]:
                envelope.append((left_line, lRange))
                i += 1
            else:
                envelope.append((right_line, rRange))
                j += 1

        # same range
            '''The above is correct and translates to both ls and lh'''

        elif lstart == rstart and lend == rend:

            theta = (lstart + lend) / 2
            if is_ls:
                x = get_ls(theta, left_line, lRange)
                y = get_ls(theta, right_line, rRange)
            else:
                x = get_lh(theta, left_line, lRange)
                y = get_lh(theta, right_line, rRange)
            if x > y:
                envelope.append((left_line, lRange))
            else:
                envelope.append((right_line, rRange))
            i += 1
            j += 1


        # intersection at the end and start
        elif lend == rstart:
            envelope.append((left_line, lRange))
            i += 1
        elif rend == lstart:
            envelope.append((right_line, rRange))
            j += 1

        # both start at an intersection
        elif rstart == lstart:

            if lend < rend:
                theta = (lstart + lend) / 2
                if is_ls:
                    x = get_ls(theta, left_line, lRange)
                    y = get_ls(theta, right_line, rRange)
                else:
                    x = get_lh(theta, left_line, lRange)
                    y = get_lh(theta, right_line, rRange)
                if x > y:
                    envelope.append((left_line, lRange))
                    renv[j] = (right_line, [lend, rend])
                i += 1
            else:

                theta = (rstart + rend) / 2
                if is_ls:
                    x = get_ls(theta, left_line, lRange)
                    y = get_ls(theta, right_line, rRange)
                else:
                    x = get_lh(theta, left_line, lRange)
                    y = get_lh(theta, right_line, rRange)
                if y > x:
                    envelope.append((right_line, rRange))
                    lenv[i] = (left_line, [rend, lend])
                j += 1


        # both end at an intersection
        elif lend == rend:
            if lstart < rstart:
                theta = (rstart + rend) / 2
                if is_ls:
                    x = get_ls(theta, left_line, lRange)
                    y = get_ls(theta, right_line, rRange)
                else:
                    x = get_lh(theta, left_line, lRange)
                    y = get_lh(theta, right_line, rRange)
                if x > y:
                    envelope.append((left_line, lRange))
                else:
                    envelope.append((left_line, [lstart, rstart]))
                    envelope.append((right_line, rRange))
            else:

                theta = (lstart + lend) / 2
                if is_ls:
                    x = get_ls(theta, left_line, lRange)
                    y = get_ls(theta, right_line, rRange)
                else:
                    x = get_lh(theta, left_line, lRange)
                    y = get_lh(theta, right_line, rRange)
                if y > x:
                    envelope.append((right_line, rRange))
                else:
                    envelope.append((right_line, [rstart, lstart]))
                    envelope.append((left_line, lRange))
            i += 1
            j += 1

        # the left range subsumes the right range Here at the checking
        elif lstart < rstart and lend > rend:

            if envelope:
                _, test1 = envelope[-1]
            theta = (rstart + rend) / 2
            if is_ls:
                x = get_ls(theta, left_line, lRange)
                y = get_ls(theta, right_line, rRange)
            else:
                x = get_lh(theta, left_line, lRange)
                y = get_lh(theta, right_line, rRange)
            if y > x:
                envelope.append((left_line, [lstart, rstart]))
                envelope.append((right_line, rRange))
                lenv[i] = (left_line, [rend, lend])
            j += 1

        # THE RIGHT RANGE SUBSUMES THE LEFT RANGE
        elif rstart < lstart and rend > lend:

            theta = (lstart + lend) / 2
            if is_ls:
                x = get_ls(theta, left_line, lRange)
                y = get_ls(theta, right_line, rRange)
            else:
                x = get_lh(theta, left_line, lRange)
                y = get_lh(theta, right_line, rRange)
            if x > y:
                envelope.append((right_line, [rstart, lstart]))
                envelope.append((left_line, lRange))
                renv[j] = (right_line, [lend, rend])
            i += 1


        elif lstart < rstart and lend < rend:

            theta = (rstart + lend) / 2

            if is_ls:
                x = get_ls(theta, left_line, lRange)
                y = get_ls(theta, right_line, rRange)
            else:
                x = get_lh(theta, left_line, lRange)
                y = get_lh(theta, right_line, rRange)
            if x > y:
                envelope.append((left_line, lRange))
                renv[j] = (right_line, [lend, rend])
            else:
                envelope.append((left_line, [lstart, rstart]))
            i += 1

        elif rstart < lstart and rend < lend:

            theta = (lstart + rend) / 2

            if is_ls:
                x = get_ls(theta, left_line, lRange)
                y = get_ls(theta, right_line, rRange)
            else:
                x = get_lh(theta, left_line, lRange)
                y = get_lh(theta, right_line, rRange)
            if y > x:
                envelope.append((right_line, rRange))
                lenv[i] = (left_line, [rend, lend])
            else:
                envelope.append((right_line, [rstart, lstart]))
            j += 1


    while i < len(lenv):
        envelope.append(lenv[i])
        i += 1


    while j < len(renv):
        envelope.append(renv[j])
        j += 1


    return envelope












def calculate_envelope(ranges, s, e, is_ls):
    if s < e:
        middle = (s + e) // 2

        lenv = calculate_envelope(ranges, s, middle, is_ls)
        renv = calculate_envelope(ranges, middle + 1, e, is_ls)
        envelope = merge_env(lenv, renv, is_ls)
        return envelope
    if s > len(ranges)-1:
        return []
    else:
        return [ranges[s]]

def merge_env(lenv, renv, is_ls):

    envelope = []
    i, j = 0, 0

    while i < len(lenv) and j < len(renv):
        left_line, lRange = lenv[i]
        right_line, rRange = renv[j]
        lstart, lend = lRange
        rstart, rend = rRange

        # no overlap at all between the two ranges
        if not overlap(lRange, rRange):
            if lRange[0] < rRange[0]:
                envelope.append((left_line, lRange))
                i += 1
            else:
                envelope.append((right_line, rRange))
                j += 1

        # same range
            '''The above is correct and translates to both ls and lh'''

        elif lstart == rstart and lend == rend:

            theta = (lstart + lend) / 2
            if is_ls:
                x = get_ls(theta, left_line, lRange)
                y = get_ls(theta, right_line, rRange)
            else:
                x = get_lh(theta, left_line, lRange)
                y = get_lh(theta, right_line, rRange)
            if x < y:
                envelope.append((left_line, lRange))
            else:
                envelope.append((right_line, rRange))
            i += 1
            j += 1


        # intersection at the end and start
        elif lend == rstart:
            envelope.append((left_line, lRange))
            i += 1
        elif rend == lstart:
            envelope.append((right_line, rRange))
            j += 1

        # both start at an intersection
        elif rstart == lstart:

            if lend < rend:
                theta = (lstart + lend) / 2
                if is_ls:
                    x = get_ls(theta, left_line, lRange)
                    y = get_ls(theta, right_line, rRange)
                else:
                    x = get_lh(theta, left_line, lRange)
                    y = get_lh(theta, right_line, rRange)
                if x < y:
                    envelope.append((left_line, lRange))
                    renv[j] = (right_line, [lend, rend])
                i += 1
            else:

                theta = (rstart + rend) / 2
                if is_ls:
                    x = get_ls(theta, left_line, lRange)
                    y = get_ls(theta, right_line, rRange)
                else:
                    x = get_lh(theta, left_line, lRange)
                    y = get_lh(theta, right_line, rRange)
                if y < x:
                    envelope.append((right_line, rRange))
                    lenv[i] = (left_line, [rend, lend])
                j += 1


        # both end at an intersection
        elif lend == rend:
            if lstart < rstart:
                theta = (rstart + rend) / 2
                if is_ls:
                    x = get_ls(theta, left_line, lRange)
                    y = get_ls(theta, right_line, rRange)
                else:
                    x = get_lh(theta, left_line, lRange)
                    y = get_lh(theta, right_line, rRange)
                if x < y:
                    envelope.append((left_line, lRange))
                else:
                    envelope.append((left_line, [lstart, rstart]))
                    envelope.append((right_line, rRange))
            else:

                theta = (lstart + lend) / 2
                if is_ls:
                    x = get_ls(theta, left_line, lRange)
                    y = get_ls(theta, right_line, rRange)
                else:
                    x = get_lh(theta, left_line, lRange)
                    y = get_lh(theta, right_line, rRange)
                if y < x:
                    envelope.append((right_line, rRange))
                else:
                    envelope.append((right_line, [rstart, lstart]))
                    envelope.append((left_line, lRange))
            i += 1
            j += 1

        # the left range subsumes the right range Here at the checking
        elif lstart < rstart and lend > rend:

            if envelope:
                _, test1 = envelope[-1]
            theta = (rstart + rend) / 2
            if is_ls:
                x = get_ls(theta, left_line, lRange)
                y = get_ls(theta, right_line, rRange)
            else:
                x = get_lh(theta, left_line, lRange)
                y = get_lh(theta, right_line, rRange)
            if y < x:
                envelope.append((left_line, [lstart, rstart]))
                envelope.append((right_line, rRange))
                lenv[i] = (left_line, [rend, lend])
            j += 1

        # THE RIGHT RANGE SUBSUMES THE LEFT RANGE
        elif rstart < lstart and rend > lend:

            theta = (lstart + lend) / 2
            if is_ls:
                x = get_ls(theta, left_line, lRange)
                y = get_ls(theta, right_line, rRange)
            else:
                x = get_lh(theta, left_line, lRange)
                y = get_lh(theta, right_line, rRange)
            if x < y:
                envelope.append((right_line, [rstart, lstart]))
                envelope.append((left_line, lRange))
                renv[j] = (right_line, [lend, rend])
            i += 1


        elif lstart < rstart and lend < rend:

            theta = (rstart + lend) / 2

            if is_ls:
                x = get_ls(theta, left_line, lRange)
                y = get_ls(theta, right_line, rRange)
            else:
                x = get_lh(theta, left_line, lRange)
                y = get_lh(theta, right_line, rRange)
            if x < y:
                envelope.append((left_line, lRange))
                renv[j] = (right_line, [lend, rend])
            else:
                envelope.append((left_line, [lstart, rstart]))
            i += 1

        elif rstart < lstart and rend < lend:

            theta = (lstart + rend) / 2

            if is_ls:
                x = get_ls(theta, left_line, lRange)
                y = get_ls(theta, right_line, rRange)
            else:
                x = get_lh(theta, left_line, lRange)
                y = get_lh(theta, right_line, rRange)
            if y < x:
                envelope.append((right_line, rRange))
                lenv[i] = (left_line, [rend, lend])
            else:
                envelope.append((right_line, [rstart, lstart]))
            j += 1


    while i < len(lenv):
        envelope.append(lenv[i])
        i += 1


    while j < len(renv):
        envelope.append(renv[j])
        j += 1


    return envelope


def overlap(left, right):
    start1, end1 = left
    start2, end2 = right

    # Check if one interval starts inside the other
    if start1 <= start2 < end1 or start1 < end2 <= end1:
        return True
    if start2 <= start1 < end2 or start2 < end1 <= end2:
        return True

    # Additionally, checking if one interval is entirely within the other
    if start1 >= start2 and end1 <= end2:
        return True
    if start2 >= start1 and end2 <= end1:
        return True

    return False

def find_xs(theta, m, c, r):
    # Calculate the discriminant of the quadratic formula part
    discriminant = -c ** 2 - 2 * c * m * r * np.cos(theta) + 2 * c * r * np.sin(theta) - m ** 2 * r ** 2 * np.cos(
        theta) ** 2 + m ** 2 * r ** 2 + m * r ** 2 * np.sin(2 * theta) - r ** 2 * np.sin(theta) ** 2 + r ** 2

    # Ensure the discriminant is non-negative for real solutions
    if discriminant < 0:
        return "The discriminant is negative, real solutions do not exist."

    # Calculate the two possible x values
    x1 = (-np.sqrt(discriminant) - c * m + m * r * np.sin(theta) + r * np.cos(theta)) / (m ** 2 + 1)
    x2 = (np.sqrt(discriminant) - c * m + m * r * np.sin(theta) + r * np.cos(theta)) / (m ** 2 + 1)

    return x1, x2


def get_lh(angle, line, ranges):
    m, c = find_equation(line)

    try:
        # Attempt to unpack the values
        x1, x2 = find_xs(angle, m, c, RADIUS)
    except ValueError as e:
        # Catch the error if too many values are returned
        full_result = find_xs(angle, m, c, RADIUS)
        print("Error: too many values to unpack. Received:", full_result)
        # Optionally, handle or re-raise the error
        print('The line is: {}, and the angle of intersection is: {}'.format(line, angle))
        print('The ranges are {}'.format(ranges))
        return 1000

    y1 = (m*x1) + c
    y2 = (m*x2) + c

    if is_point_on_line_segment(line, (x1, y1)) and is_point_on_line_segment(line, (x2, y2)):

        p1, q1, r1 = (0, 0), (RADIUS * np.cos(angle), RADIUS * np.sin(angle)), (x1, y1)
        p2, q2, r2 = (0, 0), (RADIUS * np.cos(angle), RADIUS * np.sin(angle)), (x2, y2)

        '''Orientation will return 0 if the points are colinear, 1 if the points go clockwise, and 2 if counter-clockwise. 
            This allows us to know if we want the smaller or larger angle of the three points'''

        orient = orientation(p1, q1, r1)
        if orient == 1:
            arc_of_theta1 = calculate_angle(p1, q1, r1)
        elif orient == 2:
            arc_of_theta1 = (2 * np.pi) - calculate_angle(p1, q1, r1)
        else:
            arc_of_theta1 = np.pi



        orient = orientation(p2, q2, r2)
        if orient == 1:
            arc_of_theta2 = calculate_angle(p2, q2, r2)
        elif orient == 2:
            arc_of_theta2 = (2 * np.pi) - calculate_angle(p2, q2, r2)
        else:
            arc_of_theta2 = np.pi

        return RADIUS * arc_of_theta1 if arc_of_theta1 > arc_of_theta2 else RADIUS * arc_of_theta2

    else:
        if is_point_on_line_segment(line, (x1, y1)):
            p1, q1, r1 = (0, 0), (RADIUS * np.cos(angle), RADIUS * np.sin(angle)), (x1, y1)

            orient = orientation(p1, q1, r1)
            if orient == 1:
                arc_of_theta1 = calculate_angle(p1, q1, r1)
            elif orient == 2:
                arc_of_theta1 = (2 * np.pi) - calculate_angle(p1, q1, r1)
            else:
                arc_of_theta1 = np.pi

            return RADIUS * arc_of_theta1
        else:
            p2, q2, r2 = (0, 0), (RADIUS * np.cos(angle), RADIUS * np.sin(angle)), (x2, y2)

            orient = orientation(p2, q2, r2)
            if orient == 1:
                arc_of_theta2 = calculate_angle(p2, q2, r2)
            elif orient == 2:
                arc_of_theta2 = (2 * np.pi) - calculate_angle(p2, q2, r2)
            else:
                arc_of_theta2 = np.pi

            return arc_of_theta2 * RADIUS

def get_ls(angle, line, range):
    m, c = find_equation(line)
    try:
        # Attempt to unpack the values
        x1, x2 = find_xs(angle, m, c, RADIUS)
    except ValueError as e:
        # Catch the error if too many values are returned
        full_result = find_xs(angle, m, c, RADIUS)
        print("Error: too many values to unpack. Received:", full_result)
        # Optionally, handle or re-raise the error
        print('The line is: {}, and the angle of intersection is: {}'.format(line, angle))
        print('The range is: {}'.format(range))
        return 1000
    y1 = (m*x1) + c
    y2 = (m*x2) + c

    if is_point_on_line_segment(line, (x1, y1)) and is_point_on_line_segment(line, (x2, y2)):

        p1, q1, r1 = (0, 0), (RADIUS * np.cos(angle), RADIUS * np.sin(angle)), (x1, y1)
        p2, q2, r2 = (0, 0), (RADIUS * np.cos(angle), RADIUS * np.sin(angle)), (x2, y2)

        '''Orientation will return 0 if the points are colinear, 1 if the points go clockwise, and 2 if counter-clockwise. 
            This allows us to know if we want the smaller or larger angle of the three points'''

        orient = orientation(p1, q1, r1)
        if orient == 1:
            arc_of_theta1 = calculate_angle(p1, q1, r1)
        elif orient == 2:
            arc_of_theta1 = (2 * np.pi) - calculate_angle(p1, q1, r1)
        else:
            arc_of_theta1 = np.pi



        orient = orientation(p2, q2, r2)
        if orient == 1:
            arc_of_theta2 = calculate_angle(p2, q2, r2)
        elif orient == 2:
            arc_of_theta2 = (2 * np.pi) - calculate_angle(p2, q2, r2)
        else:
            arc_of_theta2 = np.pi

        return RADIUS * arc_of_theta1 if arc_of_theta1 < arc_of_theta2 else RADIUS * arc_of_theta2

    else:
        if is_point_on_line_segment(line, (x1, y1)):
            p1, q1, r1 = (0, 0), (RADIUS * np.cos(angle), RADIUS * np.sin(angle)), (x1, y1)

            orient = orientation(p1, q1, r1)
            if orient == 1:
                arc_of_theta1 = calculate_angle(p1, q1, r1)
            elif orient == 2:
                arc_of_theta1 = (2 * np.pi) - calculate_angle(p1, q1, r1)
            else:
                arc_of_theta1 = np.pi

            return RADIUS * arc_of_theta1
        else:
            p2, q2, r2 = (0, 0), (RADIUS * np.cos(angle), RADIUS * np.sin(angle)), (x2, y2)

            orient = orientation(p2, q2, r2)
            if orient == 1:
                arc_of_theta2 = calculate_angle(p2, q2, r2)
            elif orient == 2:
                arc_of_theta2 = (2 * np.pi) - calculate_angle(p2, q2, r2)
            else:
                arc_of_theta2 = np.pi

            return arc_of_theta2 * RADIUS


def find_outside_point(tangent_point_one, tangent_point_two, point1, point2):
    return (point1, point2) if not is_point_on_line_segment((tangent_point_one, tangent_point_two), point1) else (point2, point1)


def solve_u_complex(r, x, y):
    numerator = -np.sqrt(
        4 * r ** 4 * x ** 4 + 4 * r ** 4 * x ** 2 * y ** 2 - r ** 2 * x ** 6 - 2 * r ** 2 * x ** 4 * y ** 2 - r ** 2 * x ** 2 * y ** 4) + r * x ** 2 * y + r * y ** 3
    denominator = 2 * (r ** 2 * x ** 2 + r ** 2 * y ** 2)
    u1 = numerator / denominator

    numerator = np.sqrt(
        4 * r ** 4 * x ** 4 + 4 * r ** 4 * x ** 2 * y ** 2 - r ** 2 * x ** 6 - 2 * r ** 2 * x ** 4 * y ** 2 - r ** 2 * x ** 2 * y ** 4) + r * x ** 2 * y + r * y ** 3
    denominator = 2 * (r ** 2 * x ** 2 + r ** 2 * y ** 2)
    u2 = numerator / denominator

    return u1, u2


def find_closer_to_radius(x1, y1, x, y, r):
    d1 = distance_between_points((x1, y1), (x, y))
    d2 = distance_between_points((-x1, y1), (x, y))
    return (x1, y1) if abs(r - d1) < abs(r - d2) else (-x1, y1)

def get_thetas(x, y, r):
    u1, u2 = solve_u_complex(r, x, y)
    y1, y2 = r * u1, r * u2
    x1, x2 = np.sqrt(r ** 2 - (r * u1) ** 2), np.sqrt(
        r ** 2 - (r * u2) ** 2)  # We need to calculate the correct signs for x1, and x2

    if x == 0:
        x2 = -x1
    else:
        x1, y1 = find_closer_to_radius(x1, y1, x, y, r)
        x2, y2 = find_closer_to_radius(x2, y2, x, y, r)

    angle1 = min(get_theta((x1, y1)), get_theta((x2, y2)))
    angle2 = max(get_theta((x1, y1)), get_theta((x2, y2)))
    return angle1, angle2

def does_l2_lie_on_l1(l1, l2):
    p1, p2 = l2
    return is_point_on_line_segment(l1, p1) and is_point_on_line_segment(l1, p2)

def is_point_on_line_segment(segment, point):
    ((x1, y1), (x2, y2)) = segment
    (x, y) = point

    # Check if x and y are within the bounds of the line segment
    if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
        return True
    else:
        return False


def find_equation(line):
    p1, p2 = line

    x1, y1 = p1
    x2, y2 = p2

    m = (y2 - y1) / (x2 - x1) # slope formula
    c = y2 - (m * x2)

    return m, c

def solve_u(m, c, r):
    # Pre-calculate common terms
    m2 = m ** 2  # m squared
    c2 = c ** 2  # c squared
    r2 = r ** 2  # r squared
    m2_plus_1 = m2 + 1  # m squared plus 1
    m2_plus_1_sqr = m2_plus_1 ** 2  # (m squared plus 1) squared
    m2_plus_1_cube = m2_plus_1 ** 3  # (m squared plus 1) cubed
    m2_r2 = m2 * r2  # m squared times r squared
    r4 = r2 ** 2  # r to the power 4

    term1 = (8 * c2) / (m2_plus_1_sqr * r2)
    term2 = (2 * (c2 * (-m2) - 3 * c2 + m2_r2 + r2)) / (m2_plus_1_sqr * r2)
    term3 = (2 * c2 * m2_r2 + 2 * c2 * r2 + 2 * m2 * r4 + 2 * r4) / (m2_plus_1_sqr * r4)

    denom = cmath.sqrt((4 * m2) / m2_plus_1_sqr + 4 / m2_plus_1_sqr)

    term4 = (64 * c2 * c / (m2_plus_1_cube * r * r2) +
             32 * c * (c2 - 2 * m2_r2 - r2) / (m2_plus_1_sqr * r * r2) +
             32 * c * (c2 * (-m2) - 3 * c2 + m2_r2 + r2) / (m2_plus_1_cube * r * r2)) / (
                    4 * cmath.sqrt((4 * m2) / m2_plus_1_sqr + 4 / m2_plus_1_sqr))

    c_mr = c / (m2_plus_1 * r)  # c divided by ((m squared plus one) times r)

    # Compute the solutions
    sqrt_term = cmath.sqrt(term1 + term2 - term3 - term4)
    u1 = -0.5 * sqrt_term + c_mr - (0.5 * denom)
    u2 = 0.5 * sqrt_term + c_mr - (0.5 * denom)

    sqrt_term = cmath.sqrt(term1 + term2 - term3 + term4)
    u3 = -0.5 * sqrt_term + c_mr + (0.5 * denom)
    u4 = 0.5 * sqrt_term + c_mr + (0.5 * denom)

    # Return the real part of the solution
    return [float(u.real) for u in [u1, u2, u3, u4] if u.imag == 0]

def distance_between_points(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def return_correct_x(m, c, y, possible_x, r):
    b = y - ((-1 / m) * possible_x)
    x1 = (b - c) / (m + (1 / m))
    y1 = (-1 / m) * x1 + b
    d1 = abs(distance_between_points((x1, y1), (possible_x, y)))
    b = y - ((1 / m) * possible_x)
    x2 = (b - c) / (m + (1 / m))
    y2 = (-1 / m) * x2 + b
    d2 = abs(distance_between_points((x2, y2), (-possible_x, y)))
    return (possible_x, (x1, y1)) if abs(r - d1) < abs(r - d2) else (-possible_x, (x2, y2))

def get_quadrant(point):
    x, y = point
    return QUADRANTS[(x > 0, y >= 0)]

def get_solution_ranges(line, r, i):
    m, c = find_equation(line)

    point1, point2 = line
    x1, y1 = point1
    x2, y2 = point2


    x = solve_u(m, c, r)
    sin_theta_1, sin_theta_2 = min(x), max(x)
    circle_one_center_y = r * sin_theta_1
    circle_two_center_y = r * sin_theta_2
    circle_one_center_x, tan_points1 = return_correct_x(m, c, circle_one_center_y, r * np.sqrt(1 - sin_theta_1 ** 2),
                                                        r)
    circle_two_center_x, tan_points2 = return_correct_x(m, c, circle_two_center_y, r * np.sqrt(1 - sin_theta_2 ** 2),
                                                        r)



    solution1 = np.arctan2(circle_one_center_y, circle_one_center_x)
    solution1 += 2 * np.pi if solution1 < 0 else 0
    solution2 = np.arctan2(circle_two_center_y, circle_two_center_x)
    solution2 += 2 * np.pi if solution2 < 0 else 0

    # If the y-intercept is positive
    if c > 0:
        x1, y1 = tan_points1
        x2, y2 = tan_points2
        if x2 > x1:
            temp = solution1
            solution1 = solution2
            solution2 = temp
            temp = tan_points1
            tan_points1 = tan_points2
            tan_points2 = temp


    else:
        x1, y1 = tan_points1
        x2, y2 = tan_points2
        if x2 < x1:
            temp = solution1
            solution1 = solution2
            solution2 = temp
            temp = tan_points1
            tan_points1 = tan_points2
            tan_points2 = temp

    '''circle = Circle((0, 0), RADIUS, color='b', fill=False)
    circle2 = Circle((0, 0), 2 * RADIUS, color='r', fill=False)
    circle3 = Circle((RADIUS * np.cos(solution1), RADIUS * np.sin(solution1)), RADIUS, color='r', fill=False)
    circle4 = Circle((RADIUS * np.cos(solution2), RADIUS * np.sin(solution2)), RADIUS, color='b', fill=False)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-2 * RADIUS, 2 * RADIUS)
    ax.set_ylim(-2 * RADIUS, 2 * RADIUS)
    ax.add_artist(circle)
    ax.add_artist(circle2)
    ax.add_artist(circle3)
    ax.add_artist(circle4)
    x, y = tan_points1
    plt.plot(x, y, marker='o', markersize=5, color='red')
    x, y = tan_points2
    plt.plot(x, y, marker='o', markersize=5, color='blue')
    ax.axline((0, c), slope=m, color="red", label=f'y = {m}x + {c}')
    fig.savefig('ls/ls {}.png'.format(i), dpi=300)
    plt.close()'''

    return (solution1, solution2), (tan_points1, tan_points2)

def get_ls_lh(point):
    theta_of_intersection1, theta_of_intersection2 = get_thetas(point[0], point[1], RADIUS)
    p, q, r = (0, 0), (RADIUS * np.cos(theta_of_intersection1), RADIUS * np.sin(theta_of_intersection1)), point

    '''Orientation will return 0 if the points are colinear, 1 if the points go clockwise, and 2 if counter-clockwise. 
    This allows us to know if we want the smaller or larger angle of the three points'''
    orient = orientation(p, q, r)
    if orient == 1:
        arc_of_theta1 = calculate_angle(p, q, r)
    elif orient == 2:
        arc_of_theta1 = (2 * np.pi) - calculate_angle(p, q, r)
    else:
        arc_of_theta1 = np.pi

    p, q, r = (0, 0), (RADIUS * np.cos(theta_of_intersection2), RADIUS * np.sin(theta_of_intersection2)), point
    orient = orientation(p, q, r)
    if orient == 1:
        arc_of_theta2 = calculate_angle(p, q, r)
    elif orient == 2:
        arc_of_theta2 = (2 * np.pi) - calculate_angle(p, q, r)
    else:
        arc_of_theta2 = np.pi

    return (theta_of_intersection1, theta_of_intersection2) if arc_of_theta1 < arc_of_theta2 else (theta_of_intersection2, theta_of_intersection1)

def are_the_same_point(p1, p2):
    return p1[0] == p2[0] and p1[1] == p2[1]

def intersection_query(line_segments, r):
    ls = []
    lh = []

    '''*******There are cases we did not think about, it is entirely possible that an end point is within the circle 2 * PI and another without'''

    '''All of our possible cases are going to be discussed next in the following, they should also be considered in the following order:
             1. Both of the tangent points lie on the line segment-- Here for *******
             2. The line segment lies within the two tangent points
             3. The line segment lies outside of the starting tangent point-- Here for *******
             4. The line segment lies outside of the ending tangent point-- Here for *******
             5. Only the starting tangent point lies on the line segment
             6. Only the ending tangent lies on the line segment
              '''

    both_tangents_are_on_the_line_segment = []
    only_the_starting_tangent_is_on_the_line_segment = []
    only_the_ending_tangent_is_on_the_line_segment = []
    the_line_segment_is_outside_of_the_starting_tangent_point = []
    the_line_segment_is_outside_of_the_ending_tangent_point = []
    the_line_segment_is_in_between_both_tangent_points = []


    for i, line in enumerate(line_segments):

        solutions, tangent_points = get_solution_ranges(line, r, i)

        tan1, tan2 = tangent_points

        if does_l2_lie_on_l1((tan1, tan2), line):  # if the line segment lies between tangents

            the_line_segment_is_in_between_both_tangent_points.append((line, solutions, tangent_points))
        else:
            if is_point_on_line_segment(line, tan1):
                if is_point_on_line_segment(line, tan2):

                    both_tangents_are_on_the_line_segment.append((line, solutions, tangent_points))
                else:

                    only_the_starting_tangent_is_on_the_line_segment.append((line, solutions, tangent_points))
            elif is_point_on_line_segment(line, tan2):
                only_the_ending_tangent_is_on_the_line_segment.append((line, solutions, tangent_points))

            else: # in this case the line segment has to lie outside one of the tangent points
                if distance_between_points(tan1, line[0]) < distance_between_points(tan2, line[0]):
                    the_line_segment_is_outside_of_the_starting_tangent_point.append((line, solutions, tangent_points))
                else:
                    the_line_segment_is_outside_of_the_ending_tangent_point.append((line, solutions, tangent_points))



    # Case in which both tangent points lie on the line segment, if this is the case, the range is simply defined by the start and end range for both ls and lh * not trues1
    for i, thing in enumerate(both_tangents_are_on_the_line_segment):
        line, solutions, tangent_points = thing
        solution1, solution2 = solutions
        '''Given that Both end points of a line segment are outside of the circle 2 * PI, it is correct that the range
           is completely defined from solution1 to solution2, however, very clearly we have problems if they are not.
            1. If Both of the end points of the line segment are inside of the circle 2 * PI, We will have two ranges of 
            solutions for both ls and lh. 
            2. If only the point closest to the starting tangent point is inside the circle of 2 * PI, then only ls will 
            have two ranges of solutions.
            3. If only the points closest to the ending tangent point is inside the circle of 2 * PI, then only lh will
            have two ranges of solutions.'''
        '''We can fix this later because it is not present in our current data'''
        ls.append((line, [solution1, solution2]))
        lh.append((line, [solution1, solution2]))

    #j = 0
    '''This is an easier case for the following reasons: 
        1. Both ls and lh are defined in between the two tangent points. 
        2. They are both defined from one point to the other, with no gaps. '''
    for i, thing in enumerate(the_line_segment_is_in_between_both_tangent_points):
        line, solutions, tangent_points = thing
        tan1, tan2 = tangent_points
        p1, p2 = line
        if distance_between_points(p2, tan1) < distance_between_points(p1, tan1):
            temp = p2
            p2 = p1
            p1 = temp


        '''The shorter arc by definition defines ls'''
        ls_start, lh_start = get_ls_lh(p1)
        ls_end, lh_end = get_ls_lh(p2)
        if ls_start > ls_end:
            ls.append((line, [0.0, ls_end]))
            ls.append((line, [ls_start, 2 * np.pi]))
        else:
            ls.append((line, [ls_start, ls_end]))
        if lh_start > lh_end:
            lh.append((line, [0.0, lh_end]))
            lh.append((line, [lh_start, 2 * np.pi]))
        else:
            lh.append((line, [lh_start, lh_end]))

    ''' The next two cases have precision errors that will be hard to figure out'''

    for i, thing in enumerate(only_the_starting_tangent_is_on_the_line_segment):
        line, solutions, tangent_points = thing

        solution1, solution2 = solutions
        p1, p2 = line
        tan1, tan2 = tangent_points

        '''In this case, lh will not be defined over one of the points, and ls will be defined over two
        separate contiguous ranges. One of which starts from the starting tangent solution. First, we find the point that
        lies outside of the two tangent points.'''
        point_outside, point_inside = find_outside_point(tan1, tan2, p1, p2)
        if not (are_the_same_point(tan1, point_inside)):
            '''If only the starting tangent is on the line and the starting tangent is not either of the points
            then one point is inside the two tangents, and one point is outside of the two tangents.'''
            if distance_between_points((0, 0), point_outside) < 2 * RADIUS:
                '''If the point outside of the two tangents is also inside of the circle of radius 2 * PI
                then, and only then can we break up ls this way'''
                # The theta that defines the end of the first range will be the one with the longer arc and the start of the second range will be the shorter arc
                lsStart2, lsEnd1 = get_ls_lh(point_outside)
                ls.append((line, [solution1, lsEnd1]))

                lsEnd2, lh_end = get_ls_lh(point_inside)
                ls.append((line, [lsStart2, lsEnd2]))
                #problems are here
                '''serious floating point issues are here, we are solving with the following two lines currently'''
                lh.append((line, [solution1, lh_end]))

            else:
                ''' If the point outside of the two tangents is also outside of the circle of radius 2 * PI, no range is broken up'''
                lsend, lhend = get_ls_lh(point_inside)
                ls.append([solution1, lsend])
                lh.append([solution1, lhend])
        else:
            if distance_between_points((0, 0), point_outside) < 2 * RADIUS:
                # The theta that defines the end of the first range will be the one with the longer arc and the start of the second range will be the shorter arc
                lsStart2, lsEnd1 = get_ls_lh(point_outside)
                ls.append((line, [solution1, lsEnd1]))

                lsEnd2, lh_end = get_ls_lh(point_inside)
                ls.append((line, [lsStart2, lsEnd2]))
            else:
                lsend, lhend = get_ls_lh(point_inside)
                ls.append((line, [solution1, lsend]))


    for i, thing in enumerate(only_the_ending_tangent_is_on_the_line_segment):
        line, solutions, tangent_points = thing
        solution1, solution2 = solutions
        p1, p2 = line
        tan1, tan2 = tangent_points

        '''In this case, ls will not be defined over one of the points, and lh will be defined over two
        separate contiguous range, unless the outside point lies outside of the circle of radius 2 * PI. 
        One of which starts from the starting tangent solution. First, we find the point that
        lies outside of the two tangent points.'''
        point_outside, point_inside = find_outside_point(tan1, tan2, p1, p2)
        # This could still very likely be a problem for future cases
        if not (are_the_same_point(tan2, point_inside)):
            if distance_between_points((0, 0), point_outside) < 2 * RADIUS:
                # The theta that defines the end of the first range will be the one with the longer arc and the start of the second range will be the shorter arc
                ls_start, lh_start1 = get_ls_lh(point_inside)
                ls.append((line, [ls_start, solution2]))
                lh_start2, lh_end1 = get_ls_lh(point_outside)
                lh.append((line, [lh_start1, lh_end1]))
                lh.append((line, [lh_start2, solution2]))
            else:
                ls_start, lh_start1 = get_ls_lh(point_inside)
                ls.append((line, [ls_start, solution2]))
                lh.append((line, [lh_start1, solution2]))
        else:
            if distance_between_points((0, 0), point_outside) < 2 * RADIUS:
                # The theta that defines the end of the first range will be the one with the longer arc and the start of the second range will be the shorter arc
                lhStart2, lhEnd1 = get_ls_lh(point_outside)
                lh.append((line, [lhStart2, solution2]))

                _, lh_start = get_ls_lh(point_inside)
                lh.append((line, [lh_start, lhEnd1]))
            else:
                lsend, lhstart = get_ls_lh(point_inside)
                lh.append((line, [lhstart, solution2]))

    for i, thing in enumerate(the_line_segment_is_outside_of_the_starting_tangent_point):
        line, solutions, tangent_points = thing
        p1, p2 = line
        tan1, tan2 = tangent_points

        '''For this case only ls is defined, we can identify the first point of contact by the point closest to the 
        starting tangent. If p1 is the first point of contact, we will have the longer solution to p1 and p2 making up the first range
        and the shorter solution, this time p2-p1 making up the second range. '''

        p1, p2 = (p1, p2) if distance_between_points(p1, tan1) < distance_between_points(p2, tan1) else (p2, p1)
        if np.linalg.norm(p1) > 2 * RADIUS or np.linalg.norm(p2) > 2 * RADIUS:
            end, start = get_ls_lh(p1)
            if start > end:
                ls.append((line, [start, 2 * np.pi]))
                ls.append((line, [0, end]))
            else:
                ls.append((line, [start, end]))
        else:
            range2end, range1start = get_ls_lh(p1)
            range2start, range1end = get_ls_lh(p2)
            if range1start > range1end:
                ls.append((line, [range1start, 2 * np.pi]))
                ls.append((line, [0, range1end]))
            else:
                ls.append((line, [range1start, range1end]))
            if range2start > range2end:
                ls.append((line, [range2start, 2 * np.pi]))
                ls.append((line, [0, range2end]))
            else:
                ls.append((line, [range2start, range2end]))



    for i, thing in enumerate(the_line_segment_is_outside_of_the_ending_tangent_point):
        line, solutions, tangent_points = thing
        p1, p2 = line
        tan1, tan2 = tangent_points

        p1, p2 = (p1, p2) if distance_between_points(p1, tan2) < distance_between_points(p2, tan2) else (p2, p1)
        # This could still very likely be a problem for future cases
        if np.linalg.norm(p1) > 2 * RADIUS or np.linalg.norm(p2) > 2 * RADIUS:
            end, start = get_ls_lh(p1)
            if start > end:
                lh.append((line, [start, 2 * np.pi]))
                lh.append((line, [0, end]))
            else:
                lh.append((line, [start, end]))
        else:
            lhrange2end, lhrange1start = get_ls_lh(p1)
            lhrange2start, lhrange1end = get_ls_lh(p2)
            if lhrange1start > lhrange1end:
                lh.append((line, [lhrange1start, 2 * np.pi]))
                lh.append((line, [0, lhrange1end]))
            else:
                lh.append((line, [lhrange1start, lhrange1end]))

            if lhrange2start > lhrange2end:
                lh.append((line, [lhrange2start, 2 * np.pi]))
                lh.append((line, [0, lhrange2end]))
            else:
                lh.append((line, [lhrange2start, lhrange2end]))
        '''In this case the first range will be the longer solution to p1, to the longer solution to p2. 
        The second range will be the shorter solution to p2 to the shorter solution of p1'''

    return ls, lh


# Get the polar angle of a point, return in range [0-2*pi]
def get_theta(point):
    x, y = point
    theta = np.arctan2(y, x)
    if theta < 0:
        theta += 2*np.pi
    return theta

# Sort key for list of lists
def sort_list_of_lists_by_first_element(element):
    return element[0]

# merge a list of range lists that are sorted
def merge(ranges):
    merged_ranges = []
    if ranges:
        merged_ranges = [ranges[0]]
        for start, end in ranges[1:]:
            previous_start, previous_end = merged_ranges[-1]
            if start <= previous_end:
                # the max function is used in cases where the next range is subsumed
                merged_ranges[-1] = [previous_start, max(end, previous_end)]
            else:
                merged_ranges.append([start, end])
    return merged_ranges

# return the complement of a list of ranges
def complement_list_of_range_lists(ranges):
    complement = []

    # the ranges of invisibility start further along the 0*pi radians
    if ranges[0][0] > 0:
        complement = [[0, ranges[0][0]]]

    # Complement
    for i, range in enumerate(ranges[:-1]):
        _, start = range
        end, _ = ranges[i+1]
        # we do not have to worry about empty ranges because consecutive ranges with _, end == start, _ are merged in merge
        complement.append([start, end])

    # if the last range ends before 2*pi radians, we need to add that range from starting from its end
    if ranges[-1][1] < 2*np.pi:
        start = ranges[-1][1]
        end = 2*np.pi
        complement.append([start, end])

    return complement

def is_in_range(distance, line, target):
    return find_closest_point_on_a_line_segment_and_its_distance(line, target) <= distance

def find_closest_point_on_a_line_segment_and_its_distance(line, target):
    p1, p2 = line
    vecA = np.array(p1)
    vecB = np.array(p2)
    vecC = np.array(target)

    vec1 = vecB - vecA
    vec2 = vecC - vecA

    dot_product = np.dot(vec1, vec2)
    normalizer = np.linalg.norm(vec1) ** 2

    scaler = dot_product / normalizer

    if scaler <= 0:
        closest = p1
    elif scaler >= 1:
        closest = p2
    else:
        closest = vecA + (vec1*scaler)

    return np.linalg.norm(closest)


# Intersection Query Logic
def orientation(p, q, r):
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
    if val > 0:
        return 1  # Clockwise
    elif val < 0:
        return 2  # Counterclockwise
    else:
        return 0  # Collinear

def on_segment(p, q, r):
    if q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]):
        return True
    return False

def do_intersect(seg1, seg2):
    p1, q1 = seg1
    p2, q2 = seg2

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special Cases
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True

    return False  # Doesn't fall in any of the above cases

def check_intersection(A, list_of_segments):
    for segment in list_of_segments:
        if do_intersect(A, segment):
            return True
    return False

# Calculate angle returns the smaller angle at AB and BC
def calculate_angle(A, B, C):
    # Convert points to numpy arrays
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)

    # Create vectors AB and BC
    AB = A - B
    BC = C - B

    # Calculate the dot product and the magnitudes of AB and BC
    dot_product = np.dot(AB, BC)
    magnitude_AB = np.linalg.norm(AB)
    magnitude_BC = np.linalg.norm(BC)

    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude_AB * magnitude_BC)

    # Calculate the angle in radians
    angle_radians = np.arccos(cos_angle)

    return angle_radians

'''# code for plotting circles and line segments
circle = Circle((0, 0), RADIUS, color='b', fill=False)
circle1 = Circle((0, 0), 300, color='r', fill=False)
fig, ax = plt.subplots()
lines = LineCollection(line_segments, colors='black', linewidths=2)
ax.set_aspect('equal')
ax.set_xlim(-300, 300)
ax.set_ylim(-300, 300)
ax.add_artist(circle)
ax.add_artist(circle1)
ax.add_collection(lines)
plt.show()'''

times = []
print('Total line segments: {}'.format(len(line_segments)))

# Outer Radial Visibility Finding
pre_non_config_time = timeit.default_timer()
ranges_of_invisibility_on_outer_circle = []
for line in line_segments:
    p1, p2 = line
    theta1 = get_theta(p1)
    theta2 = get_theta(p2)
    min_theta = min(theta1, theta2)
    max_theta = max(theta1, theta2)

    # we have to check here that our ranges aren't reversed, we are doing a check for the line segment crossing the positive x axis
    # Determine if there is a wrapping around the circle
    s, p = find_equation((p1, p2))
    if np.sign(p1[1]) != np.sign(p2[1]) and (np.sign(p1[1]) + np.sign(p2[1]) <=0) and p/s > 0:
        ranges_of_invisibility_on_outer_circle.append([max_theta, 2 * np.pi])
        if min_theta > 0:
            ranges_of_invisibility_on_outer_circle.append([0, min_theta])
    else:
        # Normal range addition without wrapping
        ranges_of_invisibility_on_outer_circle.append([min_theta, max_theta])

# Sort the ranges before the merge
sorted_ranges_before_merge = sorted(ranges_of_invisibility_on_outer_circle, key=sort_list_of_lists_by_first_element)


#merge the sorted ranges of invisibility
merged_sorted_ranges_of_invisibility = merge(sorted_ranges_before_merge)


'''# code for plotting circles and line segments and the arcs that are NOT VISIBLE on the OUTER MOST CIRCLE
circle = Circle((0, 0), RADIUS, color='b', fill=False)
circle1 = Circle((0, 0), 300, color='r', fill=False)
fig, ax = plt.subplots()
for segment in line_segments:
    ax.plot(*zip(*segment), color='k')
ax.set_aspect('equal')
ax.add_artist(circle)
ax.add_artist(circle1)
for range in merged_sorted_ranges_of_invisibility:
    start, end = range
    arc = Wedge((0, 0), 300, start * 180 / np.pi, end * 180 / np.pi, width=0.1 * 300, color='yellow',
                alpha=0.5)
    ax.add_artist(arc)
ax.set_xlim(-300, 300)
ax.set_ylim(-300, 300)
fig.savefig('Ranges of Invisiblity on Outer Circle.png', dpi=600)
plt.show()'''

# We need to get the complement of the invisible ranges
outer_visible_ranges = complement_list_of_range_lists(merged_sorted_ranges_of_invisibility)
post_non_config_time = timeit.default_timer()
times.append(post_non_config_time - pre_non_config_time)
print('All non-articulated solutions found in average of {} seconds.'.format(sum(times)/len(times)))

print(len(outer_visible_ranges))

'''# code for plotting circles and line segments and the arcs that ARE VISIBLE on the OUTER MOST CIRCLE
circle = Circle((0, 0), RADIUS, color='b', fill=False)
circle1 = Circle((0, 0), 300, color='r', fill=False)
fig, ax = plt.subplots()
for segment in line_segments:
    ax.plot(*zip(*segment), color='k')
ax.set_aspect('equal')
ax.add_artist(circle)
ax.add_artist(circle1)
for range in outer_visible_ranges:
    start, end = range
    arc = Wedge((0, 0), 300, start * 180 / np.pi, end * 180 / np.pi, width=0.1 * 300, color='yellow',
                alpha=0.5)
    ax.add_artist(arc)
ax.set_xlim(-300, 300)
ax.set_ylim(-300, 300)
fig.savefig('Ranges of Visiblity on Outer Circle.png', dpi=600)
plt.show()'''
times = []

pre_art_time = timeit.default_timer()
# Filter out all line segments that are not inside the center most circle of radius = length(bc)
lines_inside_center_circle = []
for line in line_segments:
    if is_in_range(RADIUS, line, (0, 0)):
        lines_inside_center_circle.append(line)

'''# code for plotting circles and line segments inside of the center most circle
circle = Circle((0, 0), RADIUS, color='b', fill=False)
circle1 = Circle((0, 0), 300, color='r', fill=False)
fig, ax = plt.subplots()
lines = LineCollection(lines_inside_center_circle, colors='black', linewidths=2)
ax.set_aspect('equal')
ax.set_xlim(-300, 300)
ax.set_ylim(-300, 300)
ax.add_artist(circle)
ax.add_artist(circle1)
ax.add_collection(lines)
plt.show()'''

# Outer Radial Visibility Finding
ranges_of_invisibility_on_inner_circle = []
for line in lines_inside_center_circle:
    p1, p2 = line
    theta1 = get_theta(p1)
    theta2 = get_theta(p2)
    min_theta = min(theta1, theta2)
    max_theta = max(theta1, theta2)
    # we have to check here that our ranges aren't reversed, we are doing a check for the line segment crossing the positive x axis
    if np.sign(p1[1]) != np.sign(p2[1]) and (p1[0] > 0 or p2[0] > 0) and ((np.sign(p1[1]) + np.sign(p2[1])) <= 0):
        ranges_of_invisibility_on_inner_circle.append([max_theta, 2*np.pi])
        if min_theta > 0:
            ranges_of_invisibility_on_inner_circle.append([0, min_theta])
    else:
        ranges_of_invisibility_on_inner_circle.append([min_theta, max_theta])

sorted_ranges_before_merge_of_invisibility_on_inner_circle = sorted(ranges_of_invisibility_on_inner_circle, key=sort_list_of_lists_by_first_element)
merged_ranges_of_invisibility_on_inner_circle = merge(sorted_ranges_before_merge_of_invisibility_on_inner_circle)
ranges_of_visibility_on_inner_circle = complement_list_of_range_lists(merged_ranges_of_invisibility_on_inner_circle)

'''# code for plotting circles and line segments and the arcs that ARE VISIBLE on the INNER MOST CIRCLE
circle = Circle((0, 0), RADIUS, color='b', fill=False)
circle1 = Circle((0, 0), 300, color='r', fill=False)
fig, ax = plt.subplots()
for segment in line_segments:
    ax.plot(*zip(*segment), color='k')
ax.set_aspect('equal')
ax.add_artist(circle)
ax.add_artist(circle1)
for range in ranges_of_visibility_on_inner_circle:
    start, end = range
    arc = Wedge((0, 0), 30, start * 180 / np.pi, end * 180 / np.pi, width=0.1 * 300, color='yellow',
                alpha=0.5)
    ax.add_artist(arc)
ax.set_xlim(-300, 300)
ax.set_ylim(-300, 300)
fig.savefig('Ranges of Visiblity on Inner Circle.png', dpi=600)
plt.show()'''

# We now need to get the extremal Points of visibility on the inner circle
extremal_points_on_inner_circle = []
for range in ranges_of_visibility_on_inner_circle:
    start, end = range
    extremal_points_on_inner_circle.append((RADIUS * np.cos(start), RADIUS * np.sin(start)))
    extremal_points_on_inner_circle.append((RADIUS * np.cos(end), RADIUS * np.sin(end)))

# We now need to find the ranges of visibility relative to our extremal points on the inner circle
valid_intermediate_configurations  = []
entry_points_from_outer_circle = {}
for i, point in enumerate(extremal_points_on_inner_circle):
    vector_of_extremal = np.array(point)
    # For each line segment we will vectorize it and subtract our point from it
    ranges_of_visibility_from_extremal_points = []
    for line in line_segments:
        p1, p2 = line
        vector1 = np.array(p1)
        vector2 = np.array(p2)

        new_vector1 = vector1 - vector_of_extremal
        new_vector2 = vector2 - vector_of_extremal

        theta1 = get_theta(new_vector1)
        theta2 = get_theta(new_vector2)

        min_theta = min(theta1, theta2)
        max_theta = max(theta1, theta2)


        # we have to check here that our ranges aren't reversed, we are doing a check for the line segment crossing the positive x axis
        if np.sign(p1[1]) != np.sign(p2[1]) and (p1[0] > 0 or p2[0] > 0) and ((np.sign(p1[1]) + np.sign(p2[1])) <= 0):  # Different signs in y-component
            ranges_of_visibility_from_extremal_points.append([max_theta, 2 * np.pi])
            if min_theta > 0:
                ranges_of_visibility_from_extremal_points.append([0, min_theta])
        else:
            ranges_of_visibility_from_extremal_points.append([min_theta, max_theta])


    sorted_ranges_of_external_invisibility_from_extremal_point = sorted(ranges_of_visibility_from_extremal_points, key=sort_list_of_lists_by_first_element)
    merged_invisibility_from_extremal = merge(sorted_ranges_of_external_invisibility_from_extremal_point)
    visible_ranges_from_extremal_point = complement_list_of_range_lists(merged_invisibility_from_extremal)

    # code for plotting circles and line segments and the arcs that ARE VISIBLE on the OUTER MOST CIRCLE From extremal point
    '''circle = Circle((0, 0), RADIUS, color='b', fill=False)
    circle1 = Circle((0, 0), 300, color='r', fill=False)
    fig, ax = plt.subplots()
    for segment in line_segments:
        ax.plot(*zip(*segment), color='k')
    ax.set_aspect('equal')
    ax.add_artist(circle)
    ax.add_artist(circle1)
    for range in visible_ranges_from_extremal_point:
        start, end = range
        arc = Wedge((0, 0), 300, start * 180 / np.pi, end * 180 / np.pi, width=0.1 * 300, color='yellow',
                    alpha=0.5)
        ax.add_artist(arc)
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    x, y = point
    plt.plot(x, y, 'ro')
    fig.savefig('Ranges of Visiblity on Outer Circle from extremal point {}.png'.format(i), dpi=600)
    plt.show()'''

    # We now need to get all points on the outer circle, that make up part of an articulated path
    possible_points_A_for_extremal = []
    for range in visible_ranges_from_extremal_point:
        start, end = range
        A = np.array((300 * np.cos(start), 300 * np.sin(start)))
        possible_points_A_for_extremal.append((A[0], A[1]))
        A = np.array((300 * np.cos(end), 300 * np.sin(end)))
        possible_points_A_for_extremal.append((A[0], A[1]))

    # We will now get the points C of every A point, this is where the intermediate configuration should end before rotating
    for j, point in enumerate(possible_points_A_for_extremal):
        A = np.array(point)
        BA = vector_of_extremal - A
        unit_BA = BA / np.linalg.norm(BA)
        vector_AC = unit_BA * 30
        C = vector_of_extremal + vector_AC
        A = (A[0], A[1])
        B = (vector_of_extremal[0], vector_of_extremal[1])
        C = (C[0], C[1])

        '''# code for plotting circles intermediate configurations before elimination
        circle = Circle((0, 0), RADIUS, color='b', fill=False)
        circle1 = Circle((0, 0), 300, color='r', fill=False)
        fig, ax = plt.subplots()
        for segment in line_segments:
            ax.plot(*zip(*segment), color='k')
        ax.set_aspect('equal')
        ax.add_artist(circle)
        ax.add_artist(circle1)

        x, y = A
        plt.plot(x, y, 'ro')
        x, y = B
        plt.plot(x, y, 'ro')
        x, y = C
        plt.plot(x, y, 'ro')

        ax.set_xlim(-300, 300)
        ax.set_ylim(-300, 300)

        fig.savefig('Intermediate Configurate for B point {} and A point {}.png'.format(i, j), dpi=600)
        plt.show()'''


        segment = (B, C)
        if not check_intersection(segment, line_segments):
            '''# code for plotting circles intermediate configurations after elimination for BC Intersection
            circle = Circle((0, 0), RADIUS, color='b', fill=False)
            circle1 = Circle((0, 0), 300, color='r', fill=False)
            fig, ax = plt.subplots()
            for segment in line_segments:
                ax.plot(*zip(*segment), color='k')
            ax.set_aspect('equal')
            ax.add_artist(circle)
            ax.add_artist(circle1)

            x, y = A
            plt.plot(x, y, 'ro')
            x, y = B
            plt.plot(x, y, 'ro')
            x, y = C
            plt.plot(x, y, 'ro')

            ax.set_xlim(-300, 300)
            ax.set_ylim(-300, 300)

            fig.savefig('Intermediate Configurate for B point {} and A point {} after Elimination.png'.format(i, j), dpi=600)
            plt.show()'''

            rotation = calculate_angle((0, 0), B, C)

            if rotation <= np.pi/2:
                valid_intermediate_configurations.append((A, B, C))
                '''# code for plotting circles intermediate configurations after elimination for BC Intersection
                circle = Circle((0, 0), RADIUS, color='b', fill=False)
                circle1 = Circle((0, 0), 300, color='r', fill=False)
                fig, ax = plt.subplots()
                for segment in line_segments:
                    ax.plot(*zip(*segment), color='k')
                ax.set_aspect('equal')
                ax.add_artist(circle)
                ax.add_artist(circle1)

                x, y = A
                plt.plot(x, y, 'ro')
                x, y = B
                plt.plot(x, y, 'ro')
                x, y = C
                plt.plot(x, y, 'ro')

                ax.set_xlim(-300, 300)
                ax.set_ylim(-300, 300)

                fig.savefig('Intermediate Configurate for B point {} and A point {} after Elimination.png'.format(i, j), dpi=600)
                plt.show()'''
post_art = timeit.default_timer()
times.append((post_art-pre_art_time))
print('Average time to find {} Extremal Articulated Paths was {} seconds'.format(len(valid_intermediate_configurations), sum(times)/len(times)))

# Now we will gather every line segment that is within 2r from the target point
line_segments_within_2r = []
for line in line_segments:
    if is_in_range(2 * RADIUS, line, (0, 0)):
        line_segments_within_2r.append(line)


ls, lh = intersection_query(line_segments_within_2r, RADIUS)




ls_lower_envelope_cc = calculate_envelope(ls, 0, len(ls)-1, True)
lh_lower_envelope_cc = calculate_envelope(lh, 0, len(lh)-1, False)
ls_lower_envelope_clock = calculate_envelopecc(lh, 0, len(lh)-1, False)
lh_lower_envelope_clock = calculate_envelopecc(ls, 0, len(ls)-1, True)


'''lowers = [ls_lower_envelope_cc, lh_lower_envelope_cc, ls_lower_envelope_clock, lh_lower_envelope_clock]
folder_names = ['ls_lower_envelope_CC', 'lh_lower_envelope_CC', 'ls_lower_envelope', 'lh_lower_envelope']

for j, lower in enumerate(lowers):
    for i, thing in enumerate(lower):
    
        line, ranges = thing
        start, end = ranges
        p1, p2 = line
        x1, y1 = p1
        x2, y2 = p2
    
        circle = Circle((0, 0), RADIUS, color='b', fill=False)
        circle1 = Circle((0, 0), 300, color='r', fill=False)
        circle2 = Circle((RADIUS * np.cos(start), RADIUS * np.sin(start)), RADIUS, color='r', fill=False)
        circle3 = Circle((RADIUS * np.cos(end), RADIUS * np.sin(end)), RADIUS, color='b', fill=False)
        fig, ax = plt.subplots()
        lines = LineCollection(line_segments, colors='black', linewidths=2)
        ax.set_aspect('equal')
        ax.set_xlim(-300, 300)
        ax.set_ylim(-300, 300)
        ax.add_artist(circle)
        ax.add_artist(circle1)
        ax.add_artist(circle2)
        ax.add_artist(circle3)
        ax.add_collection(lines)
        plt.plot(x1, y1, 'ro')
        plt.plot(x2, y2, 'ro')
        fig.savefig('{}\{}.png'.format(folder_names[j], i), dpi=300)
        plt.close()'''


# we now have to figure out if the valid and invalid articulations need to be solved with
counter_clock_wise_intermediate_configs = []
clock_wise_intermediate_configs = []
for config in valid_intermediate_configurations:
    a, b, c = config
    orient = orientation(b, (0, 0), c)
    if orient == 1:
        clock_wise_intermediate_configs.append(config)
    if orient == 2:
        counter_clock_wise_intermediate_configs.append(config)

'''Full Testing Area for doing arc intersection queries'''
def binary_search_on_lower_envelopes(lower_envelope, polar_angle_of_b):
    place = len(lower_envelope) // 2

    while True:
        thing = lower_envelope[place]
        line, range = thing
        start, end = range
        if start <= polar_angle_of_b <= end:
            return thing
        else:
            if place == 0 or place == len(lower_envelope)-1:
                return None
            elif end < polar_angle_of_b < lower_envelope[place+1][1][0]:
                return None
            elif polar_angle_of_b < start:
                place = (place-1) // 2
            else:
                place = ((place+1) + (len(lower_envelope)-1)) // 2


still_valid_configs = []

for config in clock_wise_intermediate_configs:
    a, b, c = config
    theta = get_theta(b)
    thing = binary_search_on_lower_envelopes(ls_lower_envelope_clock, theta)
    if not thing:
        thing = binary_search_on_lower_envelopes(lh_lower_envelope_clock, theta)
        if not thing:
            still_valid_configs.append(config)
            continue

    line, range = thing
    m, d = find_equation(line)
    x1, x2 = find_xs(theta, m, d, RADIUS)
    y1 = (m * x1) + d
    y2 = (m * x2) + d

    angle_at_b_to_c = calculate_angle((0, 0), b, c)
    orient = orientation((0, 0), b, c)
    if orient == 1:
        angle_at_b_to_c = (2 * np.pi) - angle_at_b_to_c
    if is_point_on_line_segment(line, (x1, y1)) and is_point_on_line_segment(line, (x2, y2)):

        angle_at_b_to_intersection1 = calculate_angle((0, 0), b, (x1, y1))
        orient = orientation((0, 0), b, (x1, y1))
        if orient == 1:
            angle_at_b_to_intersection1 = (2 * np.pi) - angle_at_b_to_intersection1

        angle_at_b_to_intersection2 = calculate_angle((0, 0), b, (x1, y1))
        orient = orientation((0, 0), b, (x1, y1))
        if orient == 1:
            angle_at_b_to_intersection2 = (2 * np.pi) - angle_at_b_to_intersection2

        angle_at_b_to_intersection = min(angle_at_b_to_intersection1, angle_at_b_to_intersection2)
    else:
        if is_point_on_line_segment(line, (x1, y1)):
            angle_at_b_to_intersection = calculate_angle((0, 0), b, (x1, y1))
            orient = orientation((0, 0), b, (x1, y1))
            if orient == 1:
                angle_at_b_to_intersection = (2 * np.pi) - angle_at_b_to_intersection
        else:
            angle_at_b_to_intersection = calculate_angle((0, 0), b, (x2, y2))
            orient = orientation((0, 0), b, (x2, y2))
            if orient == 1:
                angle_at_b_to_intersection = (2 * np.pi) - angle_at_b_to_intersection

    if angle_at_b_to_c < angle_at_b_to_intersection:
        still_valid_configs.append(config)

for config in counter_clock_wise_intermediate_configs:
    a, b, c = config
    theta = get_theta(b)
    thing = binary_search_on_lower_envelopes(ls_lower_envelope_cc, theta)
    if not thing:
        thing = binary_search_on_lower_envelopes(lh_lower_envelope_cc, theta)
        if not thing:
            still_valid_configs.append(config)
            continue

    line, range = thing
    m, d = find_equation(line)
    x1, x2 = find_xs(theta, m, d, RADIUS)
    y1 = (m * x1) + d
    y2 = (m * x2) + d

    angle_at_b_to_c = calculate_angle((0, 0), b, c)
    orient = orientation((0, 0), b, c)
    if orient == 2:
        angle_at_b_to_c = (2 * np.pi) - angle_at_b_to_c
    if is_point_on_line_segment(line, (x1, y1)) and is_point_on_line_segment(line, (x2, y2)):

        angle_at_b_to_intersection1 = calculate_angle((0, 0), b, (x1, y1))
        orient = orientation((0, 0), b, (x1, y1))
        if orient == 2:
            angle_at_b_to_intersection1 = (2 * np.pi) - angle_at_b_to_intersection1

        angle_at_b_to_intersection2 = calculate_angle((0, 0), b, (x1, y1))
        orient = orientation((0, 0), b, (x1, y1))
        if orient == 2:
            angle_at_b_to_intersection2 = (2 * np.pi) - angle_at_b_to_intersection2

        angle_at_b_to_intersection = min(angle_at_b_to_intersection1, angle_at_b_to_intersection2)
    else:
        if is_point_on_line_segment(line, (x1, y1)):
            angle_at_b_to_intersection = calculate_angle((0, 0), b, (x1, y1))
            orient = orientation((0, 0), b, (x1, y1))
            if orient == 2:
                angle_at_b_to_intersection = (2 * np.pi) - angle_at_b_to_intersection
        else:
            angle_at_b_to_intersection = calculate_angle((0, 0), b, (x2, y2))
            orient = orientation((0, 0), b, (x2, y2))
            if orient == 2:
                angle_at_b_to_intersection = (2 * np.pi) - angle_at_b_to_intersection

    if angle_at_b_to_c < angle_at_b_to_intersection:
        still_valid_configs.append(config)

def compare_points(point1, point2):
    x, y = point1
    theta1 = np.arctan2(y, x)
    if theta1 < 0:
        theta1 += 2 * np.pi

    x2, y2 = point2
    theta2 = np.arctan2(y2, x2)
    if theta2 < 0:
        theta2 += 2 * np.pi

    if theta1 < theta2:
        return -1  # Point1 comes before Point2
    elif theta1 > theta2:
        return 1   # Point1 comes after Point2
    else:
        # Angles are the same, sort by distance from origin
        if (x**2 + y**2) < (x2**2 + y2**2):
            return -1
        elif (x**2 + y**2) > (x2**2 + y2**2):
            return 1
        else:
            return 0

def return_direction(p1, p2, direction):
    x1, y1 = p1
    x2, y2 = p2
    m, c = find_equation((p1, p2))
    midx = (x1+x2)/2
    midy = (y1+y2)/2
    slope = -1/m
    y_intercept = midy - (slope*midx)
    possible1_x = midx + 1
    possible2_x = midx - 1
    possible1_y = (slope*possible1_x) + y_intercept
    possible2_y = (slope*possible2_x) + y_intercept
    if orientation(p1, (midx, midy), (possible1_x, possible1_y)) == direction:
        return ((possible1_x - midx), (possible1_y - midy))
    return ((possible2_x - midx), (possible2_y - midy))

def binary_search_on_extremes(direction, node):
    ulist = node.upper_catalog
    llist = node.lower_catalog
    uextreme = 0
    lextreme = 0

    if ulist[0] >= direction:
        uextreme = ulist[0]
    else:
        start, end = 0, len(ulist)
        while start < end:
            mid = (start + end) // 2
            if ulist[mid] >= direction and (mid == 0 or ulist[mid-1] < direction):
                uextreme = ulist[mid]
                break
            else:
                if ulist[mid] < direction:
                    start = mid + 1
                else:
                    end = mid

    if llist[0] >= direction:
        lextreme = llist[0]
    else:
        start, end = 0, len(llist)
        while start < end:
            mid = (start + end) // 2
            if llist[mid] >= direction and (mid == 0 or llist[mid-1] < direction):
                lextreme = llist[mid]
                break
            else:
                if llist[mid] < direction:
                    start = mid + 1
                else:
                    end = mid

    return node.upper_extremes[uextreme], node.lower_extremes[lextreme]

'''Full Static Convex Hull Testing Area'''
third_valid_check = []
points = []
for line in line_segments_within_2r:
    p1, p2 = line
    points += [p1, p2]
points = sorted(points, key=cmp_to_key(compare_points))
print(len(points))
hull_tree, root = build_tree(points)
hull_treefc, rootfc = build_treefc(points)

#start = time.time()
for configuration in still_valid_configs:

    A, B, C = configuration
    thetaB = get_theta(B)
    thetaC = get_theta(C)

    hulls = []
    if 1.5 * np.pi <= thetaB <= 2 * np.pi and 0 <= thetaC <= .5 * np.pi:
        hulls += get_hulls(root, thetaB, 2 * np.pi)
        hulls += get_hulls(root, 0, thetaC)
    elif 1.5 * np.pi <= thetaC <= 2 * np.pi and 0 <= thetaB <= .5 * np.pi:
        hulls += get_hulls(root, thetaC, 2 * np.pi)
        hulls += get_hulls(root, 0, thetaB)
    else:
        hulls += get_hulls(root, min(thetaB, thetaC), max(thetaB, thetaC))

    x1, y1 = B
    x2, y2 = C
    midx = (x1 + x2) / 2
    midy = (y1 + y2) / 2
    m, c = find_equation((B, C))
    slope = (-1) / m
    b = y1 - (slope * x1)
    p1 = (midx + 5, (slope * (midx + 5)) + b)
    p2 = (midx - 5, (slope * (midx - 5)) + b)
    closest_point = p1 if np.linalg.norm(p1) < np.linalg.norm(p2) else p2
    x, y = closest_point
    vector = (x - midx, y - midy)
    query_direction = get_theta(vector)



    is_empty = True
    for node in hulls:
        p1, p2 = binary_search_on_extremes(query_direction, node)
        if orientation(B, C, p1) == orientation(B, C, (0, 0)) or orientation(B, C, p2) == orientation(B, C, (0, 0)):
            is_empty = False
            break
    if is_empty:
        third_valid_check.append(configuration)

print(root.upper_catalog)
print(rootfc.upper_catalog)
print(len(rootfc.upper_catalog))
print(len(set(rootfc.upper_catalog))+1)
for i, thing in enumerate(rootfc.upper_catalog):
    if i < len(rootfc.upper_catalog)-1:
        if thing > rootfc.upper_catalog[i+1]:
            print('This is out of order')
            print(thing)
            print(rootfc.upper_catalog[i+1])


