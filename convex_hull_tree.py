from mriToLines import run
import matplotlib.pyplot as plt
import os
from functools import cmp_to_key
import numpy as np
import matplotlib.pyplot as plt
import os

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

def orientation(p, q, r):
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
    if val > 0:
        return 1  # Clockwise
    elif val < 0:
        return 2  # Counterclockwise
    else:
        return 0  # Collinear


def find_equation(line):
    p1, p2 = line

    x1, y1 = p1
    x2, y2 = p2

    m = (y2 - y1) / (x2 - x1) # slope formula
    c = y2 - (m * x2)

    return m, c

def get_theta(point):
    x, y = point
    theta = np.arctan2(y, x)
    if theta < 0:
        theta += 2*np.pi
    return theta


class node:

    def __init__(self, points, start, end, leaf_pointers):
        self.parent = None
        self.left_child = None
        self.right_child = None
        self.min = points[start]
        self.max = points[end]
        self.points_sorted_by_x = []
        self.upper_convex_hull = []
        self.lower_convex_hull = []
        self.upper_catalog = []
        self.lower_catalog = []
        self.upper_extremes = {}
        self.lower_extremes = {}
        self.build_node(points, start, end, leaf_pointers)


    def build_node(self, points, start, end, leaf_pointers):
        if start == end:
            self.points_sorted_by_x = [points[start]]
            self.upper_convex_hull.append(points[start])
            self.lower_convex_hull.append(points[start])
            self.upper_catalog = [2 * np.pi]
            self.lower_catalog = [2 * np.pi]
            self.lower_extremes = {2 * np.pi: points[start]}
            self.upper_extremes = {2 * np.pi: points[start]}
            leaf_pointers[points[start]] = self
        else:
            mid = (end + start) // 2
            self.left_child = node(points, start, mid, leaf_pointers)
            self.right_child = node(points, mid+1, end, leaf_pointers)
            left_list = self.left_child.points_sorted_by_x
            right_list = self.right_child.points_sorted_by_x

            # Do the merge method on left childs list and right childs list to have a list sorted by x followed by y
            i = 0
            j = 0
            while i < len(left_list) and j < len(right_list):
                if self.is_less_than(left_list[i], right_list[j]):
                    self.points_sorted_by_x.append(left_list[i])
                    i += 1
                else:
                    self.points_sorted_by_x.append(right_list[j])
                    j += 1
            while i < len(left_list):
                self.points_sorted_by_x.append(left_list[i])
                i += 1
            while j < len(right_list):
                self.points_sorted_by_x.append(right_list[j])
                j += 1

            # build the convex hulls using Andrews monotone chain
            self.lower_convex_hull, self.upper_convex_hull = convex_hull(self.points_sorted_by_x)

            for place, p1 in enumerate(self.upper_convex_hull[:-1]):
                p2 = self.upper_convex_hull[place + 1]
                if p1[0] == p2[0] or p1[1] == p2[1]:
                    continue
                direction = get_theta(return_direction(p1, p2, 1))
                if self.upper_catalog and self.upper_catalog[-1] == direction:
                    continue
                else:
                    self.upper_catalog.append(direction)
                    self.upper_extremes[direction] = p1
            self.upper_catalog.append(2 * np.pi)
            self.upper_extremes[2 * np.pi] = self.upper_convex_hull[-1]

            for place, p1 in enumerate(self.lower_convex_hull[:-1]):
                p2 = self.lower_convex_hull[place + 1]
                if p1[0] == p2[0] or p1[1] == p2[1]:
                    continue
                direction = get_theta(return_direction(p1, p2, 1))
                if self.lower_catalog and self.lower_catalog[-1] == direction:
                    continue
                else:
                    self.lower_catalog.append(direction)
                    self.lower_extremes[direction] = p1
            self.lower_catalog.append(2 * np.pi)
            self.lower_extremes[2 * np.pi] = self.lower_convex_hull[-1]



            #plot_points_and_hull_to_file(self.points_sorted_by_x, self.convex_hull_of_descendents, 'convex_hulls', '{}_{}_to_{}_{}.png'.format(self.points_sorted_by_x[0][0], self.points_sorted_by_x[0][1], self.points_sorted_by_x[-1][0], self.points_sorted_by_x[-1][1]))
            #print(self.convex_hull_of_descendents)



    def is_less_than(self, element1, element2):
        x1, y1 = element1
        x2, y2 = element2
        if x1 < x2:
            return True
        elif x1 == x2:
            return y1 < y2
        return False


# Andrews Monotone Chain Algorithm borrowed from wikipedia
def convex_hull(points):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list.
    return lower, upper

def build_tree(points):
    leaf_pointers = {}
    root = node(points, 0, len(points)-1, leaf_pointers)
    return leaf_pointers, root


def plot_points_and_hull_to_file(points, hull, file_path, file_name):
    """
    Plot the set of points and the convex hull and save to a file.

    :param points: List of tuples (x, y) representing the points.
    :param hull: List of tuples (x, y) representing the points in the convex hull.
    :param file_path: String representing the path to the folder where the file will be saved.
    :param file_name: String representing the name of the file.
    """
    plt.figure()  # Create a new figure

    # Create a scatter plot of the points
    plt.scatter([p[0] for p in points], [p[1] for p in points], color='blue', label='Points')

    # Draw the convex hull
    if hull:
        hull.append(hull[0])  # Ensure the hull is closed by repeating the first point at the end
        plt.plot([p[0] for p in hull], [p[1] for p in hull], color='red', label='Convex Hull')

    # Plot a circle of radius 60 centered at the origin
    circle = plt.Circle((0, 0), 60, color='green', fill=False, linestyle='--', label='Circle of radius 60')
    plt.gca().add_patch(circle)

    # Adding labels and legend
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Plot of Points, Convex Hull, and Circle')
    plt.legend()

    # Check if the directory exists, if not, create it
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Save the plot as a PNG file
    plt.savefig(os.path.join(file_path, file_name))
    plt.close()  # Close the plot to free up memory

def get_hulls(root, lower, upper):
    nodes_with_hulls = []
    split_node = find_split_node(root, lower, upper)
    node = split_node
    if is_leaf(node):
        if lower <= get_theta(node.points_sorted_by_x[0]) <= upper:
            nodes_with_hulls.append(node)
    else:
        node = node.left_child
        while not is_leaf(node):
            if lower <= get_theta(node.left_child.max):
                nodes_with_hulls.append(node.right_child)
                node = node.left_child
            else:
                node = node.right_child
        if lower <= get_theta(node.points_sorted_by_x[0]) <= upper:
            nodes_with_hulls.append(node)

        node = split_node.right_child
        while not is_leaf(node):
            if upper >= get_theta(node.right_child.min):
                nodes_with_hulls.append(node.left_child)
                node = node.right_child
            else:
                node = node.left_child
        if lower <= get_theta(node.points_sorted_by_x[0]) <= upper:
            nodes_with_hulls.append(node)
    return nodes_with_hulls



def find_split_node(root, lower, upper):
    node = root

    while not is_leaf(node) and (upper <= get_theta(node.left_child.max) or lower > get_theta(node.left_child.max)):
        if upper <= get_theta(node.left_child.max):
            node = node.left_child
        else:
            node = node.right_child
    return node

def is_leaf(node):
    return node.left_child is None and node.right_child is None