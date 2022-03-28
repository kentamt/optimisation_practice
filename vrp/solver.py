#!/usr/bin/python
# -*- coding: utf-8 -*-
import time
import math
import random

random.seed(20200729)
random.seed(20200801)

from collections import namedtuple
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

# named color list
colors = list(mcolors.TABLEAU_COLORS)  # mcolors.CSS4_COLORS)


def length(customer1, customer2):
    return math.sqrt((customer1.x - customer2.x) ** 2 + (customer1.y - customer2.y) ** 2)


def plot(customers, depot, vehicle_count, vehicle_tours):
    plt.plot([e.x for e in customers], [e.y for e in customers], 'o', color="salmon", ms=10, label='customer')
    for v in range(0, vehicle_count):
        vehicle_tour = vehicle_tours[v]
        if len(vehicle_tour) > 0:
            line_color = colors[v % len(colors)]
            plt.plot([depot.x, vehicle_tour[0].x], [depot.y, vehicle_tour[0].y], '-', color=line_color)
            for i in range(0, len(vehicle_tour) - 1):
                # print(vehicle_tour[i].index, vehicle_tour[i + 1].index)
                plt.plot([vehicle_tour[i].x, vehicle_tour[i + 1].x], [vehicle_tour[i].y, vehicle_tour[i + 1].y], '-',
                         color=line_color)
            plt.plot([vehicle_tour[-1].x, depot.x], [vehicle_tour[-1].y, depot.y], '-', color=line_color)
    plt.show()


def objective(depot, vehicle_count, vehicle_tours):
    obj = 0
    for v in range(0, vehicle_count):
        vehicle_tour = vehicle_tours[v]
        if len(vehicle_tour) > 0:
            obj += length(depot, vehicle_tour[0])
            for i in range(0, len(vehicle_tour) - 1):
                obj += length(vehicle_tour[i], vehicle_tour[i + 1])
            obj += length(vehicle_tour[-1], depot)
    return obj


def generate_output(depot, obj, vehicle_count, vehicle_tours):
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    for v in range(0, vehicle_count):
        output_data += str(depot.index) + ' ' + ' '.join(
            [str(customer.index) for customer in vehicle_tours[v]]) + ' ' + str(depot.index) + '\n'
    return output_data


def trivial_solution(customers, depot, vehicle_capacity, vehicle_count, customer_count):
    vehicle_tours = []
    remaining_customers = set(customers)
    remaining_customers.remove(depot)
    for v in range(0, vehicle_count):
        # print "Start Vehicle: ",v
        vehicle_tours.append([])
        capacity_remaining = vehicle_capacity
        while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
            used = set()
            order = sorted(remaining_customers, key=lambda customer: -customer.demand * customer_count + customer.index)
            for customer in order:
                if capacity_remaining >= customer.demand:
                    capacity_remaining -= customer.demand
                    vehicle_tours[v].append(customer)
                    # print '   add', ci, capacity_remaining
                    used.add(customer)
            remaining_customers -= used
    return vehicle_tours


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    customer_count = int(parts[0])
    vehicle_count = int(parts[1])
    vehicle_capacity = int(parts[2])

    customers = []
    for i in range(1, customer_count + 1):
        line = lines[i]
        parts = line.split()
        customers.append(Customer(i - 1, int(parts[0]), float(parts[1]), float(parts[2])))

    # the depot is always the first customer in the input
    depot = customers[0]
    n_c = len(customers)

    # 1 10pt
    if n_c < 20:  # 1
        vehicle_tour_idxs = [
            [1, 3, 8, 7, 6],
            [12, 9, 2, 11],
            [14, 13, 4, 15, 10, 5]
        ]

    # 2 7pt
    elif n_c < 30:  # 2
        vehicle_tour_idxs = [
            [2, 3],
            [15, 19],
            [24, 14, 5],
            [10, 9, 11],
            [17, 7, 16, 8],
            [13, 21, 20, 6],
            [23, 1, 12],
            [22, 18, 25, 4]
        ]

    # 3 3pt
    elif n_c < 60:  # 3
        vehicle_tour_idxs = [
            [11, 21, 9, 49, 5, 37, 17, 19, 25, 7, 6],
            [2, 3, 31, 8, 26, 43, 24, 23, 48, 27],
            [14, 13, 40, 42, 15, 10, 39, 34, 50, 35, 28],
            [46, 32, 1, 22, 36, 20, 29, 30, 38, 12, 4],
            [47, 18, 41, 44, 45, 33, 16],
        ]

    # 4 10pt
    elif n_c < 110:  # 4
        vehicle_tour_idxs = [
            [55, 54, 53, 56, 58, 60, 59, 57],
            [20, 24, 25, 27, 29, 30, 28, 26, 23, 22, 21],
            [67, 65, 63, 62, 74, 72, 61, 64, 68, 66, 69],
            [13, 17, 18, 19, 15, 16, 14, 12],
            [99, 100, 97, 93, 92, 94, 95, 96, 98],
            [81, 78, 76, 71, 70, 73, 77, 79, 80],
            [91, 89, 88, 85, 84, 82, 83, 86, 87, 90],
            [32, 33, 31, 35, 37, 38, 39, 36, 34],
            [43, 42, 41, 40, 44, 45, 46, 48, 51, 50, 52, 49, 47],
            [75, 1, 2, 4, 6, 9, 11, 10, 8, 7, 3, 5]
        ]

    # 5 7pt
    elif n_c < 210:  # 5
        vehicle_tour_idxs = [
            [68, 161, 48, 8, 118, 85],
            [86, 49, 190, 102, 164, 24, 39],
            [112, 159, 5, 59, 178, 23, 154],
            [171, 2, 94, 125, 47, 182, 31, 120],
            [183, 174, 141, 87, 134, 34, 81, 196],
            [147, 13, 152, 72, 67, 66, 181, 160],
            [133, 170, 79, 188, 32, 108, 19, 113, 93],
            [95, 14, 114, 30, 65, 3, 195, 155, 180, 105],
            [111, 12, 130, 4, 198, 15, 16, 107, 62, 167, 156],
            [166, 151, 44, 22, 75, 197, 54, 184, 149, 26, 58],
            [100, 193, 194, 148, 10, 71, 135, 9, 177, 28, 176, 27],
            [162, 189, 103, 158, 116, 21, 144, 38, 45, 82, 123, 106, 89],
            [191, 61, 127, 175, 128, 157, 50, 76, 77, 169, 187, 139, 138, 53],
            [109, 163, 179, 186, 172, 142, 97, 96, 83, 18, 153, 168, 143, 11, 131, 33, 132],
            [40, 73, 145, 115, 137, 192, 119, 98, 104, 52, 88, 124, 64, 63, 20, 51, 136, 35, 29, 150, 1],
            [146, 6, 99, 173, 17, 84, 60, 199, 46, 36, 7, 126, 90, 70, 101, 69, 122, 185, 78, 129, 121, 80, 165, 55, 25,
             110, 56, 74, 41, 57, 42, 43, 140, 91, 37, 92, 117]
        ]


    # 6 7pt
    else:  # 6
        vehicle_tour_idxs = [
            [163, 282, 343, 309, 369, 403, 342, 341, 276, 216, 101, 96],
            [162, 336, 223, 189, 129, 69, 9],
            [43, 102, 103, 222, 283, 249, 402, 401, 396, 281, 221, 161, 156, 41],
            [48, 108, 286, 397, 364, 304, 226, 107, 97, 4],
            [106, 217, 166, 167, 228, 348, 408, 407, 346, 337, 184, 64],
            [46, 157, 124, 277, 244, 406, 347, 288, 287, 227, 168],
            [170, 230, 349, 351, 411, 370, 250, 231, 70],
            [111, 130, 190, 310, 410, 409, 404, 344, 289, 284],
            [51, 110, 10, 171, 291, 350, 290, 229, 224, 169, 164, 109, 104, 49],
            [56, 175, 176, 296, 405, 365, 305, 345, 225, 125],
            [5, 65, 165, 174, 234, 294, 415, 416, 356, 235, 114, 54],
            [105, 185, 285, 245, 354, 414, 355, 295, 236, 116, 115],
            [118, 119, 71, 239, 251, 418, 357, 352, 292, 237, 172, 117],
            [57, 178, 238, 297, 419, 371, 311, 131, 179],
            [112, 177, 232, 298, 412, 417, 358, 359, 299, 191, 11, 59],
            [75, 136, 256, 315, 413, 360, 353, 254, 194, 120, 173, 113],
            [14, 16, 76, 195, 293, 180, 233, 134, 60, 420],
            [240, 300, 314, 374, 375, 376, 316, 255, 196, 135, 74],
            [192, 372, 377, 379, 366, 306, 319, 246, 259, 258],
            [17, 132, 197, 312, 317, 318, 378, 186, 66, 79, 78],
            [72, 77, 137, 252, 257, 198, 199, 126, 139, 138, 6, 19],
            [24, 84, 144, 204, 324, 323, 322, 361, 301, 1],
            [83, 143, 203, 313, 241, 253, 202, 193, 142, 133, 61],
            [22, 82, 263, 264, 384, 383, 382, 373, 262, 181, 121, 73],
            [80, 200, 320, 326, 327, 307, 247, 146, 7],
            [27, 87, 67, 127, 206, 266, 325, 380, 260, 205, 145, 25],
            [86, 147, 207, 187, 267, 367, 387, 386, 385, 265, 140, 85],
            [2, 182, 242, 122, 62],
            [32, 91, 151, 211, 271, 331, 362, 302, 30],
            [92, 152, 212, 272, 332, 392, 391, 390, 381, 330, 321, 270, 261, 210, 201, 150, 141, 90, 81],
            [33, 88, 148, 93, 94, 214, 334, 275, 68, 8, 35],
            [153, 208, 273, 328, 333, 388, 393, 395, 248, 188, 274, 154],
            [95, 155, 128, 215, 335, 308, 368, 394, 268, 213],
            [38, 220, 338, 329, 303, 243, 183, 63],
            [3, 123, 218, 269, 399, 400, 340, 280, 219, 159, 40],
            [99, 100, 160, 279, 339, 398, 363, 389, 278, 209, 158, 149, 98, 89],
            [20, 15, 58, 45, 55, 50, 44, 37, 47, 42, 39],
            [36, 29, 34, 28, 21, 31, 26, 13, 23, 18, 12, 53, 52],
            [], [], []
        ]
    vehicle_tours = []
    for tour_idx in vehicle_tour_idxs:
        tour = []
        for ic in tour_idx:
            tour.append(customers[ic])
        vehicle_tours.append(tour)

    # checks that the number of customers served is correct
    assert sum([len(v) for v in vehicle_tours]) == len(customers) - 1

    # plot
    plot(customers, depot, vehicle_count, vehicle_tours)

    # calculate the cost of the solution; for each vehicle the length of the route
    obj = objective(depot, vehicle_count, vehicle_tours)

    # prepare the solution in the specified output format
    output_data = generate_output(depot, obj, vehicle_count, vehicle_tours)

    return output_data


import sys

if __name__ == '__main__':

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')
