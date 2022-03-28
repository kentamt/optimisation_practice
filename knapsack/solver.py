#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from collections import namedtuple
import itertools
import pulp

Item = namedtuple("Item", ['index', 'value', 'weight'])

def solve_it_org(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    # a trivial algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    value = 0
    weight = 0
    taken = [0]*len(items)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


def solve_it_bound_1(input_data):
    # parse the input
    lines = input_data.split('\n')
    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])
    items = []
    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    # sorted by value density
    items = sorted(items, key=lambda x:float(x.value)/float(x.weight), reverse=True)

    # すべてを使った推定値, これより大きくなることはない
    estimate = sum([e.value for e in items])
    # estimate = linear_relaxization


    n = len(items) 
    taken = [0] * len(items)
        
    # bit全探索
    # for i,e in enumerate(itertools.product([0,1], repeat=n)):
    #     print(e)
    
    for i in range(2**n):  # 空集合を除く場合はrange(1, 2**len(list1))
        list2 = []
        for j in range(n):
            if (i >> j) &1 == 1:
                list2.append(1)
            else:
                list2.append(0)
        print(list2)

    # 各アイテムを使うか使わないかを枝刈りしていく
    # 深さ有線探索
    weight = 0
    value = 0
    # taken = [0] * len(items)

    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data

def solve_it_greedy(input_data):
    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    # sorted by value density
    items = sorted(items, key=lambda x:float(x.value)/float(x.weight), reverse=True)

    # a trivial algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    value = 0
    weight = 0
    taken = [0]*len(items)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


def solve_it_org(input_data):
    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))


    n = len(items)
    if n < 400:

        # DP table
        dp_table = np.zeros((capacity+1, len(items)+1), dtype=np.int)
        # print(dp_table.shape)
        for ic in range(dp_table.shape[1]):

            if ic == 0:
                continue

            # 隣の列をコピー
            dp_table[:, ic] = dp_table[:, ic-1]
            item = items[ic-1] # 列の数 = アイテムの数 + 1

            # 各行について処理
            for ir in range(dp_table.shape[0]):

                # 隣の列のir = now_capの値とitem.valueの和を候補のひとつとする
                candidate_1 = 0
                if ic > 1:
                    if ir-item.weight >= 0:
                        candidate_1 = dp_table[ir-item.weight, ic-1] + item.value # CAUTION: インデックスでバグる
                        weight = item.weight + items[ic-2].weight
                
                # 今のアイテムだけつかう
                candidate_2 = item.value
                weight = item.weight
                if weight > ir:
                    candidate_2 = 0            
                
                # 横の値をもう一つの候補とする
                candidate_3 = dp_table[ir, ic-1]
                
                # 候補同士を比較して大きい方をテーブルに代入する
                dp_table[ir, ic] = max([candidate_1, candidate_2, candidate_3])
            
        print(dp_table)
        
        # Trace back
        taken = [0]*len(items)
        ir = dp_table.shape[0]-1 # 末尾の数字
        for ic in range(1, dp_table.shape[1])[::-1]: # 逆順に列をすすめ、1まですすめる
            # print(ir, ic)
            item = items[ic-1] # 列の数 = アイテムの数 + 1
            score = dp_table[ir, ic] # 末端のスコア
            next_score = dp_table[ir, ic-1] # 隣のスコア
            if score == next_score:
                # print("same value")
                taken[ic-1] = 0
            else:
                # print("Different value")
                taken[ic-1] = 1
                
                for iir in range(0, ir)[::-1]: # irから逆順
                    if dp_table[iir, ic-1] == score - item.value:
                        ir = iir
                        break
        value = dp_table[-1, -1]
    else:
        
        # sorted by value density
        items = sorted(items, key=lambda x:float(x.value)/float(x.weight), reverse=True)

        # a trivial algorithm for filling the knapsack
        # it takes items in-order until the knapsack is full
        value = 0
        weight = 0
        taken = [0]*len(items)

        for item in items:
            if weight + item.weight <= capacity:
                taken[item.index] = 1
                value += item.value
                weight += item.weight

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data
    

def solve_it(input_data):
    lines = input_data.split('\n')
    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])
    items = []
    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))
    n_i = len(items)

    V = {}
    for i in range(n_i):
        V[i] = items[i].value

    W = {}
    for i in range(n_i):
        W[i] = items[i].weight

    problem = pulp.LpProblem("MIP", pulp.LpMaximize)

    # variables
    x = {}
    for i in range(n_i):
        x[i] = pulp.LpVariable(f"x({i})", 0, 1, pulp.LpInteger)

    # objective
    problem += pulp.lpSum(V[i] * x[i] for i in range(n_i)), "value"

    # constraints
    problem += pulp.lpSum(W[i] * x[i] for i in range(n_i)) <= capacity, "c1"
    print(problem)

    solver = pulp.COIN_CMD(maxSeconds=1000, threads=16)
    # solver = pulp.COIN_CMD(threads=16)
    result_status = problem.solve(solver)

    print("Problem")
    print(f"-" * 8)
    print(problem)
    print(f"-" * 8)
    print("")

    print("Result")
    print(f"*" * 8)
    print(f"Optimality = {pulp.LpStatus[result_status]}, ", end="")
    print(f"Objective = {pulp.value(problem.objective)}, ", end="")

    taken = [0] * n_i
    value = 0
    weight = 0
    for i in range(n_i):
        if x[i].value() == 1:
            taken[i] = 1
            value += items[i].value
            weight += items[i].weight

    # prepare the solution in the specified output format
    print(f"{capacity=}, we")
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        # print(solve_it(input_data))
        # print(solve_it_bound_1(input_data))
        print(solve_it_mip(input_data))
        # print(solve_it_dp(input_data))
        # print(solve_it_greedy(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

