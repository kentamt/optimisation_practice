#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import time
import math
import random
random.seed(20200729)
random.seed(20200801)
import copy
import time
from collections import namedtuple
import itertools
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, Tuple
import pulp
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from simanneal import Annealer
from tqdm import tqdm

Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

# named color list
colors = list(mcolors.TABLEAU_COLORS)  # mcolors.CSS4_COLORS)


def check_capacity_constraint(vehicle_tours, vehicle_capacity):
    for tour in vehicle_tours:
        total_demand = 0
        for c in tour:
            total_demand += c.demand
            if total_demand > vehicle_capacity:
                return False

    return True


class TSP(Annealer):
    """Test annealer with a travelling salesman problem."""

    def __init__(self, state, distance_matrix, points):
        self.distance_matrix = distance_matrix
        self.points = points
        super(TSP, self).__init__(state)  # important!
        self.nodeCount = len(self.state)
        self.node_list = list(range(0, self.nodeCount))

    def move(self):
        """ 2-Opt """
        tnode1 = random.randint(0, self.nodeCount - 1)
        tnode2 = random.randint(0, self.nodeCount - 1)
        sub_solution = self.state[tnode1:tnode2 + 1]
        self.state[tnode1:tnode2 + 1] = sub_solution[::-1]

    def energy(self):
        """Calculates the length of the route."""
        energy = self.distance_matrix[self.state[-1]][self.state[0]]
        for index in range(0, self.nodeCount - 1):
            energy += self.distance_matrix[self.state[index]][self.state[index + 1]]

        return energy


class VRP(Annealer):
    def __init__(self, state, depot, customers, vehicle_count, capacity, distance_matrix, tps_schedule, tps_update):
        # self.distance_matrix = distance_matrix # あるほうが早い
        super(VRP, self).__init__(state)  # important!

        # VRP
        self.capacity = capacity
        self.customers = customers
        self.depot = depot
        self.vehicle_count = vehicle_count
        self.distance_matrix = distance_matrix

        # TSP schedule
        self.schedule = tps_schedule  # {'tmax': 1000.0, 'tmin': 0.1, 'steps': 10000, 'updates': 100}
        self.tsp_period = tps_update  #
        self.recursive_max = 30
        self.iter_count = 0



    def move(self):

        state_copy = copy.deepcopy(self.state)
        # self.show_tours(self.state)

        i = 0
        if self.random_insert(i, state_copy):
            self.state = copy.deepcopy(state_copy)
        else:
            pass
            # print("Max recursive count!")


        # TSP
        if self.iter_count % self.tsp_period == 0:
            for i, tour in enumerate(self.state):
                solution = [e.index for e in tour]
                solution.insert(0, 0)
                if len(solution) > 3:
                    tsp = TSP(solution, self.distance_matrix, self.customers)
                    tsp.set_schedule(self.schedule)
                    tsp.copy_strategy = "slice"
                    solution, _ = tsp.anneal()
                    tour = [self.customers[ic] for ic in solution if not ic == 0]
                    self.state[i] = tour
        self.iter_count += 1
        # self.show_tours(self.state)
        # print("------")
        # time.sleep(0.5)
        # exit()
    def random_insert(self, i, state_copy):
        # print(f"Recursive: {i}")
        if i > self.recursive_max:
            return False
        # 0でないツアーをランダムに1つ選択
        # さらに1つ客を選ぶ
        iv1, tour1 = self.select_tour_random(state_copy)
        c1, ic1 = self.select_customer_random(tour1)
        # tour1でないランダムなツアーを1つえらぶ
        while True:
            iv2 = random.randint(0, self.vehicle_count - 1)
            if iv2 != iv1:
                break
        tour2 = state_copy[iv2]
        # tour2の走行距離が最も短くなる場所にc1を挿入
        self.insert_customer_greedy(c1, tour2)
        # c1をtour1から削除
        tour1.pop(ic1)
        # 制約条件を満たしているか確認
        is_capacity_ok = check_capacity_constraint(state_copy, self.capacity)
        if not is_capacity_ok:
            # time.sleep(0.3)
            # print("--------")
            # self.show_tours(state_copy)
            return self.random_insert(i + 1, state_copy)
        else:
            # print("Capacity constraint is OK")
            return True

    def insert_customer_greedy(self, c1, tour2):
        # tour2_copy = copy.deepcopy(tour2)
        tour2_copy = copy.copy(tour2)
        min_l_tour2 = 1000000000000000
        insert_idx = 0
        for i in range(len(tour2)):
            tour2_copy.insert(i, c1)
            l_tour2_copy = self.cal_length_tour(tour2_copy)
            if l_tour2_copy < min_l_tour2:
                min_l_tour2 = l_tour2_copy
                insert_idx = i
        tour2.insert(insert_idx, c1)

    def show_tours(self, state_copy):
        for tour in state_copy:
            for e in tour:
                print(e.index, end=" ")
            print()

    def select_customer_random(self, tour1):
        ic1 = random.randint(0, len(tour1) - 1)  # a<= x <=b
        c1 = tour1[ic1]
        return c1, ic1

    def select_tour_random(self, state_copy):
        vidx = []
        for i, tour in enumerate(state_copy):
            if len(tour) != 0:
                vidx.append(i)
        # 0でないツアーのインデックスをランダムに取得
        iv1 = random.choice(vidx)
        tour1 = state_copy[iv1]
        return iv1, tour1

    def cal_length_tour(self, tour):
        if len(tour) == 0:
            return 0
        length = self.distance_matrix[0][tour[0].index]
        for i in range(1, len(tour) - 1):
            _c1 = tour[i]
            _c2 = tour[i + 1]
            length += self.distance_matrix[_c1.index][_c2.index]
        length += self.distance_matrix[tour[-1].index][0]
        return length

    def energy(self):
        # energy = objective(self.depot, self.vehicle_count, self.state)
        energy = self.objective(self.state)

        return energy

    def objective(self, state):
        obj = 0
        for v in range(0, self.vehicle_count):
            vehicle_tour = state[v]
            # vehicle_tour = vehicle_tours[v]
            if len(vehicle_tour) > 0:
                # obj += length(depot, vehicle_tour[0])
                obj += self.distance_matrix[0][vehicle_tour[0].index]
                for i in range(0, len(vehicle_tour) - 1):
                    # obj += length(vehicle_tour[i], vehicle_tour[i + 1])
                    obj += self.distance_matrix[vehicle_tour[i].index][vehicle_tour[i+1].index]
                # obj += length(vehicle_tour[-1], depot)
                obj += self.distance_matrix[vehicle_tour[-1].index][0]
        return obj



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


def mip_solution(customers, depot, vehicle_capacity, vehicle_count, customer_count):
    n_c = customer_count
    n_v = vehicle_count

    # Demand vector
    Demand = []
    for c in range(n_c):
        Demand.append(customers[c].demand)
    Demand = np.array(Demand)

    # Adjacent matrix
    # distance between customer and facility
    c_xy = [(c.x, c.y) for c in customers]
    dist_mat = distance.cdist(c_xy, c_xy, 'euclidean')
    A = np.zeros((n_c, n_c), dtype=np.float)
    for i in range(n_c):
        for j in range(n_c):
            A[j, i] = dist_mat[j, i]

    # ==========================
    problem = pulp.LpProblem("MIP", pulp.LpMinimize)

    # Variable
    X = []
    for j in range(n_c):
        x_list = [pulp.LpVariable(f'x{j}_{i}', 0, 1, pulp.LpInteger) for i in range(n_c)]
        X.append(x_list)
    X = np.array(X)

    # Objective:
    problem += pulp.lpSum([pulp.lpDot(a, x) for a, x in zip(A, X)])

    # Constraints:
    for i in range(n_c):
        problem += X[i, i] == 0

    for i in range(1, n_c):
        problem += pulp.lpSum(X[:, i]) == 1
    for j in range(1, n_c):
        problem += pulp.lpSum(X[j, :]) == 1

    problem += pulp.lpSum(X[:, 0]) == n_v
    problem += pulp.lpSum(X[0, :]) == n_v

    sub_tours = []
    for length in range(2, n_c):
        sub_tours += itertools.combinations(range(1, n_c), length)

    for st in sub_tours:
        arcs = []
        demand = 0
        for s in st:
            demand += customers[s].demand
        for j, i in itertools.permutations(st, 2):
            arcs.append(X[j, i])
        # print(len(st) - np.max([0, np.ceil(demand / vehicle_capacity)]))
        problem += pulp.lpSum(arcs) <= np.max([0, len(st) - np.ceil(demand / vehicle_capacity)])

    print(problem)
    solver = pulp.PULP_CBC_CMD()
    result_status = problem.solve(solver)

    # print("Problem")
    # print(f"-" * 8)
    # print(problem)
    # print(f"-" * 8)
    # print("")
    #
    # print("Result")
    # print(f"*" * 8)
    # print(f"Optimality = {pulp.LpStatus[result_status]}, ", end="")
    # print(f"Objective = {pulp.value(problem.objective)}, ", end="")

    sol = []
    for j in range(n_c):
        sol_list = [e.value() for e in X[j, :]]
        sol.append(sol_list)
        # for i in range(n_c):
        #     print(f"{X[j, i].name} = {X[j, i].value()}", end=" ")
        # print()

    sol = np.array(sol)
    tours = []
    while np.sum(sol) != 0:
        tour = []  # from 0 but not append
        j = 0
        while j < n_c:
            i = np.where(sol[j, :] == 1)[0][0]  # 複数あるうちの一番わかいもの
            if i != 0:
                tour.append(customers[i])
            sol[j, i] = 0
            if i == 0:
                break

            j = i

        # print(sol)
        tours.append(tour)
        # print(sol)
        # print(tours)
    # exit()
    return tours


def cp_solution(customers, depot, vehicle_capacity, vehicle_count):
    pass


def my_index_multi(l, x):
    return [i for i, _x in enumerate(l) if _x == x]


def kmeans_clusterting(customers):

    from sklearn.cluster import KMeans

    xy_array = np.array([
        [e.x for e in customers],
        [e.y for e in customers]
    ]).T
    clustered_idxs = KMeans(n_clusters=11, random_state=10).fit_predict(xy_array)

    # 各クラスタをひとつの客とする新しいリストの作成
    num_km_customers = np.max(clustered_idxs)
    km_customers = []

    for i in range(0, num_km_customers+1):
        idxs = my_index_multi(clustered_idxs, i)
        ith_customer = [customers[ic] for ic in idxs]
        x = np.mean([e.x for e in ith_customer])
        y = np.mean([e.y for e in ith_customer])
        d = np.sum([e.demand for e in ith_customer])
        km_customer = Customer(i, d, x, y)
        km_customers.append(km_customer)

    # plt.figure()
    # plt.scatter([e.x for e in customers], [e.y for e in customers], c=clustered_idxs)  # [e.demand for e in customers])
    # plt.scatter([e.x for e in km_customers], [e.y for e in km_customers], c="red", s=3)  # [e.demand for e in customers])
    # plt.colorbar()
    # plt.show()

    # depotを取り除く
    # km_customers.pop(0)
    print(km_customers)
    return km_customers, clustered_idxs


def analyze(customers):


    for sweeped_customers in sweep(customers, 12):
        sxyd_array = np.array([
            [e.x for e in sweeped_customers.values()],
            [e.y for e in sweeped_customers.values()],
            [e.demand for e in sweeped_customers.values()],
        ])

        xyd_array = np.array([
            [e.x for e in customers],
            [e.y for e in customers],
            [e.demand for e in customers],
        ])

        plt.figure()
        plt.scatter(xyd_array[0, :], xyd_array[1, :], c='gray')
        plt.scatter(sxyd_array[0, :], sxyd_array[1, :], c=sxyd_array[2, :])
        plt.colorbar()
        plt.scatter(xyd_array[0, 0], xyd_array[1, 0], c="red", s=40)


        plt.show()
    exit()

def sweep(customers, num):
    # 真上からangleでピザカットしてピザ１かけ内のカスタマずつかえす関数
    num = 13
    angles = np.linspace(0, 2*np.pi, num)
    da = np.radians(3)
    for i in range(len(angles)-1):
        sweeped_customer = {}
        sweeped_customer[0] = customers[0]
        a1 = angles[i]
        a2 = angles[i+1]
        # a1からa2の間の点だけを取り出す
        nearest_10_c = None # demandが10で最も近いc
        nearest_20_c = None # demandが20で最も近いc
        l_min_10 = 1000000000000
        l_min_20 = 1000000000000
        for c in customers:
            if c.index == 0:
                continue

            ac = np.arctan2(c.y, c.x) + np.pi
            if a1+da <= ac < a2+da:

                # demand30分取り除く
                l = np.sqrt(c.x ** 2 + c.y ** 2)
                if l < l_min_10 and c.demand == 10:
                    l_min_10 = l
                    nearest_10_c = c

                if l < l_min_20 and c.demand == 20:
                    l_min_20 = l
                    nearest_20_c = c

                sweeped_customer[c.index] = c
                # sweeped_customer.append(c)

        # 近い10と20は取り除く
        del sweeped_customer[nearest_10_c.index]
        del sweeped_customer[nearest_20_c.index]

        print(np.sum([e.demand for e in sweeped_customer.values()]))
        yield sweeped_customer, nearest_10_c, nearest_20_c

def expand_kmeans(km_tours, clustered_idxs, customers):
    ret_tours = []
    for tour in km_tours:
        ret_tour = []
        for c in tour:
            idxs = my_index_multi(clustered_idxs, c.index)
            ith_customer = [customers[ic] for ic in idxs]
            ret_tour.extend([e for e in ith_customer])
        ret_tours.append(ret_tour)

    return ret_tours


def ls_solution(customers, depot, vehicle_capacity, vehicle_count):
    # Local Search

    # For Q.6
    coords = [(c.x, c.y) for c in customers.values()]
    dist_mat = distance.cdist(coords, coords, 'euclidean')
    distance_matrix = {}
    for i, node in enumerate(customers.values()):
        if node not in distance_matrix.keys():
            distance_matrix[node.index] = {}
        for j, node2 in enumerate(customers.values()):
            distance_matrix[node.index][node2.index] = dist_mat[i][j]

    # init
    vehicle_tours = simple_init_tours(customers, depot, vehicle_capacity, vehicle_count)
    for tour in vehicle_tours:
        random.shuffle(tour)
    is_capacity_ok = check_capacity_constraint(vehicle_tours, vehicle_capacity)
    print(f"{is_capacity_ok=}")

    # 車両の数より少ない場合は空のリストを追加する
    num_tours = len(vehicle_tours)
    for _ in range(vehicle_count - num_tours):
        vehicle_tours.append([])

    for tour in vehicle_tours:
        random.shuffle(tour)

    # print(vehicle_tours)
    # print(objective(depot, vehicle_count, vehicle_tours))
    n_c = len(customers)

    # For Q.6
    schedule = {'tmax': 10000.0, 'tmin': 0.02, 'steps': 20000, 'updates': 2000}
    tps_schedule = {'tmax': 1000.0, 'tmin': 0.02, 'steps': 10000, 'updates': 0}
    tps_update = 100

    # Do SA
    print(schedule, len(customers))
    vrp = VRP(vehicle_tours, depot, customers, vehicle_count, vehicle_capacity, distance_matrix, tps_schedule, tps_update)
    vrp.set_schedule(schedule)
    # vrp.set_schedule(vrp.auto(minutes=0.2))
    vrp.copy_strategy = "slice"
    start = time.time()
    solution, e = vrp.anneal()
    print(f"\n\n\n{time.time() - start}[sec]")

    # TSP
    tsp_schedule = {'tmax': 10000.0, 'tmin': 0.02, 'steps': 10000, 'updates': 400}
    for i, tour in enumerate(solution):
        sol = [e.index for e in tour]
        sol.insert(0, 0)
        # print(sol)
        if len(sol) > 3:
            tsp = TSP(sol, distance_matrix, customers)
            tsp.set_schedule(tsp_schedule)
            tsp.copy_strategy = "slice"
            sol, _ = tsp.anneal()
            # print(sol)
            while True:
                sol = sol[1:] + sol[:1]
                if sol[0] == 0:
                    break
            tour = [customers[ic] for ic in sol if not ic == 0]
            solution[i] = tour
            # for e in tour:
            #     print(e.index, end=" ")

    return solution


def ls_tsp(solution, customers):

    # TSP
    coords = [(c.x, c.y) for c in customers]
    dist_mat = distance.cdist(coords, coords, 'euclidean')
    distance_matrix = {}
    for i, node in enumerate(range(len(customers))):
        if node not in distance_matrix.keys():
            distance_matrix[node] = {}
        for j, node2 in enumerate(range(len(customers))):
            distance_matrix[node][node2] = dist_mat[i][j]

    tsp_schedule = {'tmax': 50000.0, 'tmin': 0.1, 'steps': 40000, 'updates': 400}
    for i, tour in enumerate(solution):
        sol = [e.index for e in tour]
        sol.insert(0, 0)

        if len(sol) > 3:
            tsp = TSP(sol, distance_matrix, customers)
            tsp.set_schedule(tsp_schedule)
            tsp.copy_strategy = "slice"
            sol, _ = tsp.anneal()

            while True:
                sol = sol[1:] + sol[:1]
                if sol[0] == 0:
                    break
            tour = [customers[ic] for ic in sol if not ic == 0]
            solution[i] = tour

    return solution


def simple_init_tours(customers, depot, vehicle_capacity, vehicle_count):
    customer_count = len(customers)
    vehicle_tours = []

    # For Q.6
    remaining_customers = set(customers.values())

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


def solution_to_vehicle_tours(customers, solution):
    vehicle_tours = []
    tour = []
    for idx in range(len(solution)):
        cidx = solution[idx]
        if cidx == 0:
            if idx == 0:
                tour = []
            else:
                vehicle_tours.append(tour)
                tour = []
        else:
            tour.append(customers[cidx])
    vehicle_tours.append(tour)
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

    # analyze(customers)

    # ------------------------------------------
    # build a solution
    # ------------------------------------------
    # For Q.6
    out_customers = {}
    out_customers[0] = customers[0] # depot
    vehicle_tours_list = []
    for sw_customers, c10, c20 in sweep(customers, 12):
        out_customers[c10.index] = c10
        out_customers[c20.index] = c20
        vehicle_tours = ls_solution(sw_customers, depot, vehicle_capacity, 3)
        vehicle_tours_list.append(vehicle_tours)

        # customer dict ==> list
        # sw_customers = sw_customers.values()
        # plot(sw_customers, depot, 3, vehicle_tours)

    vehicle_tours = ls_solution(out_customers, depot, vehicle_capacity, vehicle_count-12*3)
    # out_customers = out_customers.values()
    # plot(out_customers, depot, vehicle_count-12*3, vehicle_tours)
    vehicle_tours_list.append(vehicle_tours)

    new_vehicle_tours = []
    for vehicle_tours in vehicle_tours_list:
        for tour in vehicle_tours:
            new_vehicle_tours.append(tour)

    for tour in new_vehicle_tours:
        for c in tour:
            print(c.index, end=" ")
        print()
    vehicle_tours = new_vehicle_tours

    # ------------------------------------------
    # Post process
    # ------------------------------------------
    # plot(customers, depot, vehicle_count, vehicle_tours)

    # checks that the number of customers served is correct
    assert sum([len(v) for v in vehicle_tours]) == len(customers) - 1

    # calculate the cost of the solution; for each vehicle the length of the route
    obj = objective(depot, vehicle_count, vehicle_tours)

    # prepare the solution in the specified output format
    output_data = generate_output(depot, obj, vehicle_count, vehicle_tours)

    return output_data



if __name__ == '__main__':

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:

        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')
