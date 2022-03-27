#!/usr/bin/env python
__author__ = "Edi Topić"
__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "etopic@chem.pmf.hr"

import numpy as np
from skimage import feature, io
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import cdist


def points_to_lines(point_list, neighborhood_size, max_lines, shortest_line):
    # group points with distance less than neighborhood_size into a "line", with maximum
    # number of lines equal to max_lines, and the shortest possible line size of shortest_line
    # initialization of all arrays and variables
    # counter - number of lines
    counter = 0
    point_list_size = len(point_list)
    # used- array to check if a point is already used in a line
    used = np.zeros(shape=(point_list_size), dtype=bool)
    # index - index of a point currently checked
    index = 0
    # line_list - array of coordinates with maximum line length of 200, which is plenty for most purposes
    line_list = np.zeros(shape=(2, 200, max_lines))
    # length - actual length of each line (in points)
    length = np.zeros(shape=(max_lines))
    tmp_line = []
    line_number = 0
    temp_length = 0
    # calculate euclidian distance between each point
    distance_list = cdist(point_list, point_list)

    # number of lines cannot be greater than number of points
    while counter < (point_list_size - 1):
        # if current index does not point to any actual point, the work is done
        if index == (-1):
            break
        # if the image is too *sketchy* stop
        if line_number > max_lines:
            break
        # append the indexed coordinate to a temporary line, increase the length and set the coordinate as used
        tmp_line.append(point_list[index])
        temp_length += 1
        used[index] = True
        # find all points closer than neighborhood_size
        next_index_list = np.where(distance_list[index, :] < neighborhood_size)
        next_index = -1
        # set the next index of a first non-used point
        if not next_index_list == None:
            for a in next_index_list[0]:
                if not used[a]:
                    next_index = a
                    break
        counter += 1

        if next_index < 0:
            # if there is no next point save the line
            if len(tmp_line) > 3:
                tmp_array = np.asarray(tmp_line)
                line_list[:, :, line_number] = np.transpose(tmp_array[np.linspace(0, tmp_array.shape[0] - 1, 200, dtype=int)])
                length[line_number] = temp_length
                temp_length = 0
                line_number += 1
            tmp_line = []
            # find the next appropriate point for line start
            what_next = np.where(used == False)
            if not what_next == None:
                index = what_next[0][0]
            else:
                index = -1
        else:
            # if there is a next point then go for it in the next iteration
            index = next_index
    # there are possibly smarter approaches for next index finding but here it is
    # I like my algorithms like my stoves - mostly converting electricity to heat
    # yes intel google nasa I can provide my ingenuity for money
    return line_list[:, :, :line_number - 1], length[:line_number - 1]


def process(point_list, smooth=100, poly_order=2, neighborhood_size=6, max_lines=2000, shortest_line=3):
    # arrange lines from edge points
    line_list, length = points_to_lines(point_list, neighborhood_size, max_lines, shortest_line=shortest_line)
    # parameter variable
    uxy = np.linspace(0, 1, 200)
    # initialize spline tck list
    spline_list = np.ndarray(shape=(line_list.shape[2], 2), dtype=object)
    for i in range(line_list.shape[2]):
        # interpolate x and y values of lines with uxy as parametric variable and save tck-s
        spline_list[i, 0] = UnivariateSpline(uxy, line_list[0, :, i], s=smooth, k=poly_order)._eval_args
        spline_list[i, 1] = UnivariateSpline(uxy, line_list[1, :, i], s=smooth, k=poly_order)._eval_args
    # sort everything by approximate line length
    sorted_indices = length.argsort()[::-1]
    length = length[sorted_indices]
    spline_list = spline_list[sorted_indices, :]
    # return array of parametric spline tck values and approximate length of each spline
    return spline_list, length


def sketch(filename, sigma_param, smooth, poly_order, neighborhood_size, use_canny=True, max_lines=2000,
           shortest_line=3):
    # import image as grayscale
    image = io.imread(filename, as_gray=True)
    # if image is already edges only then convert it to boolean
    if use_canny is False:
        edges = np.array(image, dtype=bool)
    else:
        # else use a canny filter with false borders and wanted sigma
        edges = feature.canny(image, sigma=sigma_param, mode='constant', cval=False)
    # add another row/column of false values to remove potential border artefacts
    edges[1, :] = False
    edges[:, 1] = False
    edges[:, -2] = False
    edges[-2, :] = False
    # extract coordinates of edges
    edge_coordinates = np.transpose(np.where(edges))
    # process the coordinates
    spline_list, length = process(edge_coordinates, smooth=smooth, poly_order=poly_order,
                                  neighborhood_size=neighborhood_size, max_lines=max_lines, shortest_line=shortest_line)
    # return array of parametric spline tck values, approximate length of each spline, and image size
    return spline_list, length, image.shape[1], image.shape[0]

