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
import os

RADIUS = 30


def read_numbers_from_txt(file_path):
    # Change the file extension from .png to .txt
    txt_file_path = os.path.splitext(file_path)[0] + '.txt'

    # Read the numbers from the .txt file
    with open(txt_file_path, 'r') as file:
        line = file.readline().strip()
        numbers = line.split()
        return int(numbers[0]), int(numbers[1])

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

def find_equation(line):
    p1, p2 = line

    x1, y1 = p1
    x2, y2 = p2

    m = (y2 - y1) / (x2 - x1) # slope formula
    c = y2 - (m * x2)

    return m, c

def sort_list_of_lists_by_first_element(element):
    return element[0]

# Get the polar angle of a point, return in range [0-2*pi]
def get_theta(point):
    x, y = point
    theta = np.arctan2(y, x)
    if theta < 0:
        theta += 2*np.pi
    return theta


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


def find_un_articulated_paths(line_segments, RADIUS, filep):
    #plot_MRI(line_segments, RADIUS, filep)
    start = timeit.default_timer()
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
        if np.sign(p1[1]) != np.sign(p2[1]) and (p1[0] > 0 or p2[0] > 0) and ((np.sign(p1[1]) + np.sign(p2[1])) <= 0):
            ranges_of_invisibility_on_outer_circle.append([max_theta, 2 * np.pi])
            if min_theta > 0:
                ranges_of_invisibility_on_outer_circle.append([0, min_theta])
        else:
            # Normal range addition without wrapping
            ranges_of_invisibility_on_outer_circle.append([min_theta, max_theta])

    # Sort the ranges before the merge
    sorted_ranges_before_merge = sorted(ranges_of_invisibility_on_outer_circle, key=sort_list_of_lists_by_first_element)

    # merge the sorted ranges of invisibility
    merged_sorted_ranges_of_invisibility = merge(sorted_ranges_before_merge)
    outer_visibility = complement_list_of_range_lists(merged_sorted_ranges_of_invisibility)
    end = timeit.default_timer()
    #plot_visibility_of_outer_circle(line_segments, RADIUS, outer_visibility, filep)
    print('{}   {}   {}  {}'.format(filep, len(line_segments), len(outer_visibility), (end-start)))


def plot_MRI(line_segments, RADIUS, filep):
    circle = Circle((0, 0), RADIUS, color='b', fill=False)
    circle1 = Circle((0, 0), 300, color='r', fill=False)
    fig, ax = plt.subplots()
    lines = LineCollection(line_segments, colors='black', linewidths=2)
    ax.set_aspect('equal')
    ax.plot(0, 0, 'ko', markersize=1)
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.add_artist(circle)
    ax.add_artist(circle1)
    ax.add_collection(lines)
    dir = 'MRI_Plots'
    fig.savefig('{}.png'.format(os.path.join(dir, os.path.splitext(filep)[0])),
                dpi=600)
    plt.show()
    plt.close(fig)


def plot_visibility_of_outer_circle(line_segments, RADIUS, visibility, filep):

    circle = Circle((0, 0), RADIUS, color='b', fill=False)
    circle1 = Circle((0, 0), 300, color='r', fill=False)
    fig, ax = plt.subplots()
    for segment in line_segments:
        ax.plot(*zip(*segment), color='k')
    ax.set_aspect('equal')
    ax.add_artist(circle)
    ax.add_artist(circle1)
    ax.plot(0, 0, 'ko', markersize=1)
    for range in visibility:
        start, end = range
        arc = Wedge((0, 0), 300, start * 180 / np.pi, end * 180 / np.pi, width=0.1 * 300, color='yellow',
                    alpha=0.5)
        ax.add_artist(arc)
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    fig.savefig('{}.png'.format(os.path.join('Unarticulated_Paths_of_Experiments', os.path.splitext(filep)[0])), dpi=600)
    plt.show()
    plt.close(fig)

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

def is_in_range(distance, line, target):
    return find_closest_point_on_a_line_segment_and_its_distance(line, target) <= distance

def find_extremal_articulated(line_segments, RADIUS, filep):
    lines_inside_center_circle = []
    for line in line_segments:
        if is_in_range(RADIUS, line, (0, 0)):
            lines_inside_center_circle.append(line)

    ranges_of_invisibility_on_inner_circle = []

    for line in lines_inside_center_circle:
        p1, p2 = line
        theta1 = get_theta(p1)
        theta2 = get_theta(p2)
        min_theta = min(theta1, theta2)
        max_theta = max(theta1, theta2)
        # we have to check here that our ranges aren't reversed, we are doing a check for the line segment crossing the positive x axis
        if np.sign(p1[1]) != np.sign(p2[1]) and (p1[0] > 0 or p2[0] > 0) and ((np.sign(p1[1]) + np.sign(p2[1])) <= 0):
            ranges_of_invisibility_on_inner_circle.append([max_theta, 2 * np.pi])
            if min_theta > 0:
                ranges_of_invisibility_on_inner_circle.append([0, min_theta])
        else:
            ranges_of_invisibility_on_inner_circle.append([min_theta, max_theta])

    sorted_ranges_before_merge_of_invisibility_on_inner_circle = sorted(ranges_of_invisibility_on_inner_circle,
                                                                        key=sort_list_of_lists_by_first_element)
    merged_ranges_of_invisibility_on_inner_circle = merge(sorted_ranges_before_merge_of_invisibility_on_inner_circle)
    ranges_of_visibility_on_inner_circle = complement_list_of_range_lists(merged_ranges_of_invisibility_on_inner_circle)

    extremal_points_on_inner_circle = []
    for range in ranges_of_visibility_on_inner_circle:
        start, end = range
        extremal_points_on_inner_circle.append((RADIUS * np.cos(start), RADIUS * np.sin(start)))
        extremal_points_on_inner_circle.append((RADIUS * np.cos(end), RADIUS * np.sin(end)))

    valid_intermediate_configurations = []
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
            if np.sign(p1[1]) != np.sign(p2[1]) and (p1[0] > 0 or p2[0] > 0) and (
                    (np.sign(p1[1]) + np.sign(p2[1])) <= 0):  # Different signs in y-component
                ranges_of_visibility_from_extremal_points.append([max_theta, 2 * np.pi])
                if min_theta > 0:
                    ranges_of_visibility_from_extremal_points.append([0, min_theta])
            else:
                ranges_of_visibility_from_extremal_points.append([min_theta, max_theta])

        sorted_ranges_of_external_invisibility_from_extremal_point = sorted(ranges_of_visibility_from_extremal_points,
                                                                            key=sort_list_of_lists_by_first_element)
        merged_invisibility_from_extremal = merge(sorted_ranges_of_external_invisibility_from_extremal_point)
        visible_ranges_from_extremal_point = complement_list_of_range_lists(merged_invisibility_from_extremal)

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

            rotation = calculate_angle((0, 0), B, C)
            if rotation <= np.pi / 2:
                valid_intermediate_configurations.append((A, B, C))

    return valid_intermediate_configurations



directory = 'MRIs_greyscale'
for filename in os.listdir(directory):
    if filename.endswith('.png'):  # Check for PNG files
        # Construct full file path
        file_path = os.path.join(directory, filename)
        targetx, targety = read_numbers_from_txt(file_path)
        line_segments = run(file_path, targetx, targety)
        art_paths_before_check = find_extremal_articulated(line_segments, RADIUS, filename)