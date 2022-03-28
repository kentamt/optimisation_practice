from collections import namedtuple
import math
from typing import Dict, Tuple
from matplotlib import pyplot as plt
import pulp

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
    plt.legend()


def solve_it_warehouse(inputs: str) -> str:
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

    # visualization
    plot(customers, facilities)
    plt.show()

    # transportation cost matrix
    n_c = len(customers)
    n_w = len(facilities)
    T: Dict[Tuple[int, int], float] = {}
    for w in range(n_w):
        for c in range(n_c):
            T[w, c] = length(facilities[w].location, customers[c].location)

    # facility setup cost
    C: Dict[int, float] = {}
    for w in range(n_w):
        C[w] = facilities[w].setup_cost

    # ----------------
    problem = pulp.LpProblem("MIP", pulp.LpMinimize)
    x = {}  # 決定変数の集合. 店
    y = {}  # 決定変数の集合. 客

    for w in range(n_w):
        for c in range(n_c):
            y[w, c] = pulp.LpVariable(f"y({w},{c})", 0, 1, pulp.LpInteger)

    for w in range(n_w):
        x[w] = pulp.LpVariable(f"x({w})", 0, 1, pulp.LpInteger)

    # Objective:
    problem += pulp.lpSum(C[w] * x[w] for w in range(n_w)) + pulp.lpSum(
        T[w, c] * y[w, c] for w in range(n_w) for c in range(n_c)), "Total cost"

    # Subject to:
    for c in range(n_c):
        problem += sum(y[w, c] for w in range(n_w)) == 1, f"Constraint_eq_{c}"

    for w in range(n_w):
        for c in range(n_c):
            problem += y[w, c] <= x[w], f"Constraint_leq_{w, c}"

    print("Problem")
    print(f"-" * 8)
    print(problem)
    print(f"-" * 8)
    print("")
    solver = pulp.PULP_CBC_CMD()
    result_status = problem.solve(solver)

    # print("Result")
    # print(f"*" * 8)
    # print(f"Optimality = {pulp.LpStatus[result_status]}, ", end="")
    # print(f"Objective = {pulp.value(problem.objective)}, ", end="")
    # print("Solution y[w, c]: ")
    # for w in range(n_w):
    #     for c in range(n_c):
    #         print(f"{y[w, c].name} = {y[w, c].value()}")
    #
    # print("Solution x[w]: ")
    # for w in range(n_w):
    #     print(f"{x[w].name} = {x[w].value()}")
    # print("")
    # print(f"*" * 8)

# visualize
    for w in range(n_w):
        for c in range(n_c):
            if y[w, c].value() == 1:
                print(f"{y[w, c].name} = {y[w, c].value()},  ", end="")
                pf = facilities[w]
                pc = customers[c]
                plt.plot([pc.location.x, pf.location.x], [pc.location.y, pf.location.y], '-', color='royalblue')

    for w in range(n_w):
        pf = facilities[w]
        if x[w].value() == 1:
            color = 'salmon'
        else:
            color = 'gray'
        plt.plot(pf.location.x, pf.location.y, 'o', color=color)

    plt.plot([e.location.x for e in customers], [e.location.y for e in customers], ".", color='royalblue',
             label="customers")
    plt.show()


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

    n_c = len(customers)
    n_w = len(facilities)

    # visualization
    plot(customers, facilities)
    plt.show()

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
    # problem += pulp.lpSum(C[w] * x[w] for w in range(n_w)) + pulp.lpSum(
    #     T[w, c] * y[w, c] for w in range(n_w) for c in range(n_c)), "Total cost"
    problem += pulp.lpSum(C[w] * y[w, c] + T[w, c] * y[w, c] for w in range(n_w) for c in range(n_c)), "Total cost"

    # Subject to:
    for c in range(n_c):
        problem += sum(y[w, c] for w in range(n_w)) == 1, f"Customer_must_go_facility({c})"

    for w in range(n_w):
        for c in range(n_c):
            problem += y[w, c] <= x[w], f"Customer_cannot_go_closed_facility({w, c})"

    # sum(demand) <= cap_f
    for w in range(n_w):
        for c in range(n_c):
            problem += sum(D[c] * y[ww, c] for ww in range(n_w)) <= Cap[w], f"Demand cannot exceed capacity({w, c})"

    print("Problem")
    print(f"-" * 8)
    print(problem)
    print(f"-" * 8)
    print("")
    solver = pulp.PULP_CBC_CMD()
    result_status = problem.solve(solver)

    print("Result")
    print(f"*" * 8)
    print(f"Optimality = {pulp.LpStatus[result_status]}, ", end="")
    print(f"Objective = {pulp.value(problem.objective)}, ", end="")
    print("Solution y[w, c]: ")
    for w in range(n_w):
        for c in range(n_c):
            print(f"{y[w, c].name} = {y[w, c].value()}")

    print("Solution x[w]: ")
    for w in range(n_w):
        print(f"{x[w].name} = {x[w].value()}")
    print("")
    print(f"*" * 8)

    # visualize
    for w in range(n_w):
        pf = facilities[w]
        plt.plot(pf.location.x, pf.location.y, 'o', color='gray')

    for w in range(n_w):
        for c in range(n_c):
            if y[w, c].value() == 1:
                print(f"{y[w, c].name} = {y[w, c].value()},  ", end="")
                pf = facilities[w]
                pc = customers[c]
                plt.plot([pc.location.x, pf.location.x], [pc.location.y, pf.location.y], '-', color='royalblue')
                plt.plot(pf.location.x, pf.location.y, 'o', color='salmon')

    # for w in range(n_w):
    #     pf = facilities[w]
    #     print(x[w].value())
    #     if x[w].value() == 1:
    #         color = 'salmon'
    #     else:
    #         color = 'gray'
    #     plt.plot(pf.location.x, pf.location.y, 'o', color=color)

    plt.plot([e.location.x for e in customers], [e.location.y for e in customers], ".", color='royalblue',
             label="customers")
    plt.show()

    # Generate solution
    solution = []
    for c in range(n_c):
        for w in range(n_w):
            if y[w, c].value() == 1:
                solution.append(w)

    used = [0]*len(facilities)
    for facility_index in solution:
        used[facility_index] = 1

    # calculate the cost of the solution
    obj = objective(customers, facilities, solution, used)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


# lowBound, upBound を指定しないと、それぞれ -無限大, +無限大 になる
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
