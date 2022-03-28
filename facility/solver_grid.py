#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import copy
import time
from typing import Dict, Tuple
import pulp
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def objective(customers, facilities, solution, used):
    obj = sum([f.setup_cost * used[f.index] for f in facilities])
    for customer in customers:
        obj += length(customer.location, facilities[solution[customer.index]].location)
    return obj


def plot(customers: object, facilities: object) -> object:
    plt.plot([e.location.x for e in facilities], [e.location.y for e in facilities], "o", color='salmon',
             label='facilities')
    plt.plot([e.location.x for e in customers], [e.location.y for e in customers], ".", color='royalblue',
             label="customers")
    # plt.legend()

def plot_org(customers: object, facilities: object) -> object:
    plt.plot([e.location.x for e in facilities], [e.location.y for e in facilities], "o", color='gray',
             label='facilities')
    plt.plot([e.location.x for e in customers], [e.location.y for e in customers], ".", color='gray',
             label="customers")
    # plt.legend()


def solve_it(inputs: str) -> str:
    # parse the input
    lines = inputs.split('\n')
    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])

    facilities = []
    for i in range(1, facility_count + 1):
        parts = lines[i].split()
        facilities.append(Facility(i - 1,
                                   float(parts[0]),
                                   int(parts[1]),
                                   Point(float(parts[2]), float(parts[3]))
                                   )
                          )

    customers = []
    for i in range(facility_count + 1, facility_count + 1 + customer_count):
        parts = lines[i].split()
        customers.append(Customer(i - 1 - facility_count,
                                  int(parts[0]),
                                  Point(float(parts[1]), float(parts[2]))
                                  )
                         )

    customers_org = copy.copy(customers)
    facilities_org = copy.copy(facilities)

    x_min = min(np.min([e.location.x for e in customers_org]), np.min([e.location.x for e in facilities_org]))-1
    y_min = min(np.min([e.location.y for e in customers_org]), np.min([e.location.y for e in facilities_org]))-1
    x_max = max(np.max([e.location.x for e in customers_org]), np.max([e.location.x for e in facilities_org]))+1
    y_max = max(np.max([e.location.y for e in customers_org]), np.max([e.location.y for e in facilities_org]))+1

    #
    # n_div = 5 # q5
    n_div = 6  # q6

    x_intervals = np.linspace(x_min, x_max, n_div)
    y_intervals = np.linspace(y_min, y_max, n_div)

    solution = [0] * len(customers_org)
    used = [0] * len(facilities_org)
    for ix in range(len(x_intervals)-1):
        for iy in range(len(y_intervals)-1):
            customers = [e for e in customers_org if (x_intervals[ix] <= e.location.x <= x_intervals[ix + 1]) and (y_intervals[iy] <= e.location.y <= y_intervals[iy + 1])]
            facilities = [e for e in facilities_org if (x_intervals[ix] <= e.location.x <= x_intervals[ix + 1]) and (y_intervals[iy] <= e.location.y <= y_intervals[iy + 1])]
            n_c = len(customers)
            n_w = len(facilities)

            # visualization
            # plot_org(customers_org, facilities_org)
            # plot(customers, facilities)
            # plt.show()

            # transportation cost matrix
            T: Dict[Tuple[int, int], float] = {}
            for w in range(n_w):
                for c in range(n_c):
                    T[w, c] = length(facilities[w].location, customers[c].location)

            # facility setup cost
            C: Dict[int, float] = {}
            for w in range(n_w):
                C[w] = facilities[w].setup_cost

            # customer's demands
            D = {}
            for c in range(n_c):
                D[c] = customers[c].demand

            # facility capacity
            Cap = {}
            for w in range(n_w):
                Cap[w] = facilities[w].capacity

            # distance between customer and facility
            c_xy = [(c.location.x, c.location.y) for c in customers]
            f_xy = [(f.location.x, f.location.y) for f in facilities]
            dist_mat = distance.cdist(f_xy, c_xy, 'euclidean')

            Dist = {}
            for w in range(n_w):
                for c in range(n_c):
                    Dist[w, c] = dist_mat[w, c]
            # max distance
            max_distance = np.max(dist_mat)
            min_demands = np.min([e.demand for e in customers])  # ある店のキャパ/全体の需要の最小値が，ある店の顧客数の上限

            # ----------------
            problem = pulp.LpProblem("MIP", pulp.LpMinimize)
            x = {}  # 決定変数の集合. 店
            y = {}  # 決定変数の集合. 客

            # Variables
            for w in range(n_w):
                for c in range(n_c):
                    y[w, c] = pulp.LpVariable(f"y({w},{c})", 0, 1, pulp.LpInteger)

            for w in range(n_w):
                x[w] = pulp.LpVariable(f"x({w})", 0, 1, pulp.LpInteger)

            # Objective:
            problem += pulp.lpSum(C[w] * x[w] for w in range(n_w)) + pulp.lpSum(
                T[w, c] * y[w, c] for w in range(n_w) for c in range(n_c)), "Total cost"

            # Subject to:
            # Customer
            for c in range(n_c):
                problem += sum(y[w, c] for w in range(n_w)) == 1, f"Customer_must_go_facility({c})"

            # Customer and Facility
            for w in range(n_w):
                for c in range(n_c):
                    problem += y[w, c] <= x[w], f"Customer_cannot_go_closed_facility({w, c})"

            # Num of facility
            for w in range(n_w):
                 problem += sum(y[w, c] for c in range(n_c) ) <= x[w] * n_c, f"Facility cannot exceed the number of customer({w, c})"

            # Capacity
            for w in range(n_w):
                problem += sum(D[c] * y[w, c] for c in range(n_c)) <= Cap[w], f"Demand cannot exceed capacity({w, c})"

            for w in range(n_w):
                problem += sum(y[w, c] for c in range(n_c)) <= Cap[
                    w] / min_demands, f"Demand per min demand cannot exceed num of assigned customers({w})"

            # Distance
            for w in range(n_w):
                for c in range(n_c):
                    # problem += Dist[w, c] * y[w, c] <= 0.5 * max(Dist[ww, c] for ww in range(n_w)), f"Distance constraint({w, c})" # works for q4
                    problem += Dist[w, c] * y[w, c] <= max(Dist[ww, c] for ww in range(n_w)), f"Distance constraint({w, c})"

            # Solve
            # solver = pulp.COIN_CMD(fracGap=0.0, maxSeconds=1000, threads=16)
            solver = pulp.COIN_CMD(threads=16, maxSeconds=300)
            result_status = problem.solve(solver)

            # visualize
            # plot_org(customers_org, facilities_org)
            # for w in range(n_w):
            #     pf = facilities[w]
            #     plt.plot(pf.location.x, pf.location.y, 'o', color='gray')
            #
            # for w in range(n_w):
            #     for c in range(n_c):
            #         if y[w, c].value() == 1:
            #             print(f"{y[w, c].name} = {y[w, c].value()},  ", end="")
            #             pf = facilities[w]
            #             pc = customers[c]
            #             plt.plot([pc.location.x, pf.location.x], [pc.location.y, pf.location.y], '-', color='royalblue')
            #             plt.plot(pf.location.x, pf.location.y, 'o', color='salmon')
            #
            # plt.plot([e.location.x for e in customers], [e.location.y for e in customers], ".", color='royalblue',
            #          label="customers")
            # plt.show()

            # Generate solution
            for c in range(n_c):
                for w in range(n_w):
                    if y[w, c].value() == 1:
                        ic = customers[c].index
                        iw = facilities[w].index
                        solution[ic] = iw

            for facility_index in solution:
                used[facility_index] = 1

            print(solution)

    n_c = len(customers_org)
    n_w = len(facilities_org)
    for w in range(n_w):
        pf = facilities_org[w]
        plt.plot(pf.location.x, pf.location.y, 'o', color='gray')

    for ic, iw in enumerate(solution):
        pf = facilities_org[iw]
        pc = customers_org[ic]
        plt.plot([pc.location.x, pf.location.x], [pc.location.y, pf.location.y], '-', color='royalblue')
        plt.plot(pf.location.x, pf.location.y, 'o', color='salmon')

    plt.plot([e.location.x for e in customers_org], [e.location.y for e in customers_org], ".", color='royalblue',
             label="customers")
    plt.show()

    # calculate the cost of the solution
    obj = objective(customers_org, facilities_org, solution, used)
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
        print('This test requires an input file.  \
        Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')
