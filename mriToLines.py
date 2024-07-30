from PIL import Image
import numpy as np
from math import cos, sin

def run(filename, x, y):
    mri = Image.open(filename)
    matrix = np.array(mri)
    lines = []
    #cell_row, cell_col = np.random.randint(0, matrix.shape[0]), np.random.randint(0, matrix.shape[1])
    def is_color(i, j):
        return matrix[i, j] > 0

    '''while is_color(cell_row, cell_col):
        cell_row, cell_col = np.random.randint(0, matrix.shape[0]), np.random.randint(0, matrix.shape[1])'''
    cell_row, cell_col = x, y

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if is_color(i, j):
                selfx, selfy = cell_col - j, cell_row - i
                self = (-selfx, selfy)
                directions = [(i, j + 1), (i + 1, j), (i + 1, j - 1), (i + 1, j + 1)]
                for dir in directions:
                    r, c = dir
                    if r < matrix.shape[0] and r >= 0 and c < matrix.shape[1] and c >= 0:
                        if is_color(r, c):
                            lines.append((self, (-(cell_col - c), cell_row - r)))
    theta = 0.241
    new_lines = []
    for line in lines:
        p1, p2 = line
        x1, y1 = p1
        newx1 = x1*cos(theta) + y1*sin(theta)
        newy1 = -x1*sin(theta) + y1*cos(theta)

        x1, y1 = p2
        newx2 = x1*cos(theta) + y1*sin(theta)
        newy2 = -x1*sin(theta) + y1*cos(theta)

        new_lines.append(((newx1, newy1), (newx2, newy2)))

    return new_lines



