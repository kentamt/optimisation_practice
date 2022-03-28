#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import copy
from collections import namedtuple
import itertools as it
import random

random.seed(20200720)
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from scipy.spatial import distance
from simanneal import Annealer
from pprint import pprint

Point = namedtuple("Point", ['x', 'y'])
viz = False


def plot(points, solution):
    plt.scatter([e.x for e in points], [e.y for e in points], s=10)
    for i in range(len(solution)):
        n1 = solution[i]
        n2 = solution[(i + 1) % len(solution)]
        plt.plot([points[n1].x, points[n2].x], [points[n1].y, points[n2].y], "-", c="gray")

    plt.show()


class TSP(Annealer):
    """Test annealer with a travelling salesman problem."""

    # pass extra data (the distance matrix) into the constructor
    def __init__(self, state, distance_matrix, points):
        self.distance_matrix = distance_matrix
        self.points = points
        super(TSP, self).__init__(state)  # important!
        self.nodeCount = len(self.state)
        self.node_list = list(range(0, self.nodeCount))

    def move(self):
        """ 2-Opt """
        # initial_energy = self.energy()
        tnode1 = random.randint(0, self.nodeCount - 1)
        tnode2 = random.randint(0, self.nodeCount - 1)
        sub_solution = self.state[tnode1:tnode2 + 1]
        self.state[tnode1:tnode2 + 1] = sub_solution[::-1]

        # tnodes = random.sample(self.node_list, 2)
        # sub_solution = self.state[tnodes[0]:tnodes[1]+1]
        # self.state[tnodes[0]:tnodes[1]+1] = sub_solution[::-1]                
        # self.state[tnode1:tnode2+1] = self.state[tnode1:tnode2+1][::-1]        

        # plot(self.points, self.state)
        # return self.energy() - initial_energy

    def energy(self):
        """Calculates the length of the route."""
        # energy = length(self.points[self.state[-1]], self.points[self.state[0]])
        energy = self.distance_matrix[self.state[-1]][self.state[0]]
        for index in range(0, self.nodeCount - 1):
            # energy += length(self.points[self.state[index]], self.points[self.state[index+1]])
            energy += self.distance_matrix[self.state[index]][self.state[index + 1]]

        return energy


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def nearest_neighbor(points, start_idx):
    solution = []  # list(range(0, nodeCount))
    points_buffer = copy.copy(points)

    sol = start_idx
    solution.append(sol)
    nodeCount = len(points)
    node_list = list(range(0, nodeCount))
    node_list.pop(sol)

    while True:
        l_min = 1e10
        sol = None

        # 全探索
        point1 = points_buffer[solution[-1]]
        arg = None
        for idx, node in enumerate(node_list):
            point2 = points_buffer[node]
            l = length(point1, point2)

            # print(l, idx)
            if l < l_min:
                l_min = l
                sol = node
                arg = idx

        # 最小のものを追加
        solution.append(sol)

        # bufferから1個すてる
        # print(node_list, sol)
        node_list.pop(arg)

        if len(node_list) == 0:
            break

    return solution


def nearest_neighbor_distance_mat(points):
    nodeCount = len(points)

    # distance matrix
    coords = [(e.x, e.y) for e in points]
    dist_mat = distance.cdist(coords, coords, 'euclidean')

    distance_matrix = {}
    for i, node in enumerate(range(nodeCount)):
        if node not in distance_matrix.keys():
            distance_matrix[node] = {}
        for j, node2 in enumerate(range(nodeCount)):
            distance_matrix[node][node2] = dist_mat[i][j]
    exit()


def sa(points, solution):
    nodeCount = len(points)

    # distance matrix
    coords = [(e.x, e.y) for e in points]
    dist_mat = distance.cdist(coords, coords, 'euclidean')

    distance_matrix = {}
    for i, node in enumerate(range(nodeCount)):
        if node not in distance_matrix.keys():
            distance_matrix[node] = {}
        for j, node2 in enumerate(range(nodeCount)):
            distance_matrix[node][node2] = dist_mat[i][j]

    tsp = TSP(solution, distance_matrix, points)
    schedule = {'tmax': 10000.0, 'tmin': 0.1, 'steps': 1000000, 'updates': 50}

    tsp.set_schedule(schedule)
    tsp.copy_strategy = "slice"
    solution, e = tsp.anneal()

    return solution


def random_init(points):
    nodeCount = len(points)
    solution = list(range(0, nodeCount))
    random.shuffle(solution)
    return solution


def first_search_2opt(points, solution, niter=4000):
    nodeCount = len(points)
    best_obj = eval(points, solution)
    is_good_swap = True
    for _ in tqdm(range(niter)):

        if not is_good_swap:
            print("No more good swap")
            break

        # random sampling
        # target_nodes = random.sample(node_list, 2)
        is_good_swap = False
        for target_nodes in it.combinations(solution, 2):

            idx1 = solution.index(target_nodes[0])
            idx2 = solution.index(target_nodes[1])
            if np.fabs(idx1 - idx2) == 1:
                continue
            # print(target_nodes)
            tmp_solution = copy.copy(solution)

            # 2-OPT 
            sub_solution = solution[target_nodes[0]:target_nodes[1] + 1]

            # reverse order
            tmp_solution[target_nodes[0]:target_nodes[1] + 1] = sub_solution[::-1]

            # calculate the length of the tour
            obj = length(points[tmp_solution[-1]], points[tmp_solution[0]])
            for index in range(0, nodeCount - 1):
                obj += length(points[tmp_solution[index]], points[tmp_solution[index + 1]])

            if obj < best_obj:
                best_obj = obj
                solution = copy.copy(tmp_solution)
                print("solution is updated!", obj)
                is_good_swap = True

                if viz:
                    plot(points, solution)
                break
    return solution


def value_search_2opt(points, solution, niter=1):
    nodeCount = len(points)

    # TODO SAと共通で使うこと
    coords = [(e.x, e.y) for e in points]
    dist_mat = distance.cdist(coords, coords, 'euclidean')
    distance_matrix = {}
    for i, node in enumerate(range(nodeCount)):
        if node not in distance_matrix.keys():
            distance_matrix[node] = {}
        for j, node2 in enumerate(range(nodeCount)):
            distance_matrix[node][node2] = dist_mat[i][j]

    is_good_swap = True
    for _ in tqdm(range(niter)):

        if not is_good_swap:
            print("No more good swap")
            break

        # random sampling
        # target_nodes = random.sample(node_list, 2)
        is_good_swap = False

        # 一番いいswapをさがす　
        obj = distance_matrix[solution[-1]][solution[0]]
        for index in range(0, nodeCount - 1):
            obj += distance_matrix[solution[index]][solution[index + 1]]
        best_obj = obj

        best_solution = None
        for target_nodes in it.combinations(solution, 2):

            idx1 = solution.index(target_nodes[0])
            idx2 = solution.index(target_nodes[1])
            if np.fabs(idx1 - idx2) == 1:
                continue

            if idx1 == 0 and idx2 == len(solution) - 1:  # 末尾
                continue

            # print(target_nodes)
            tmp_solution = copy.copy(solution)

            # 2-OPT 
            sub_solution = solution[target_nodes[0]:target_nodes[1] + 1]

            # reverse order
            tmp_solution[target_nodes[0]:target_nodes[1] + 1] = sub_solution[::-1]

            # calculate the length of the tour
            obj = distance_matrix[solution[-1]][solution[0]]
            for index in range(0, nodeCount - 1):
                obj += distance_matrix[solution[index]][solution[index + 1]]

            if obj < best_obj:
                best_obj = obj
                best_nodes = target_nodes  # 最後にこれをつかう
                best_solution = copy.copy(tmp_solution)
                # print("solution is updated!", obj)
                is_good_swap = True

        if is_good_swap:
            solution = copy.copy(best_solution)
            if viz:
                print("solution is updated!", obj)
                plot(points, solution)
    return solution


def eval(points, solution):
    nodeCount = len(points)
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount - 1):
        l = length(points[solution[index]], points[solution[index + 1]])
        obj += l
    return obj


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')
    nodeCount = int(lines[0])
    points = []

    for i in range(1, nodeCount + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    start_idx = 0
    solution = nearest_neighbor(points, start_idx)
    # solution = value_search_2opt(points, solution, 1)
    # solution = first_search_2opt(points, solution)
    solution = sa(points, solution)
    obj = eval(points, solution)

    # 可視化
    plot(points, solution)

    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))



    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')
