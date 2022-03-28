#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import copy
from collections import namedtuple
import itertools as it
import random
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

Point = namedtuple("Point", ['x', 'y'])
viz = False

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    solution = [] # range(0, nodeCount)

    # Nearest Neighborで探索する
    points_buffer = copy.copy(points)

    sol = 0
    solution.append(sol)
    node_list = list(range(0, nodeCount))
    node_list.pop(sol)

    # while True:
    #     l_min = 1e10
    #     sol = None
        
    #     # 全探索
    #     point1 = points_buffer[solution[-1]]
    #     arg = None
    #     for idx, node in enumerate(node_list):
    #         point2 = points_buffer[node]
    #         l = length(point1, point2)        

    #         # print(l, idx)
    #         if l < l_min:
    #             l_min = l
    #             sol = node               
    #             arg = idx
                
    #     # 最小のものを追加
    #     solution.append(sol)    
        
    #     # bufferから1個すてる
    #     # print(node_list, sol)
    #     node_list.pop(arg)
            
    #     if len(node_list)==0:
    #         break


    # solution = list(range(0, nodeCount))
    solution = list(range(0, nodeCount))
    random.shuffle(solution)

    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        l = length(points[solution[index]], points[solution[index+1]])
        obj += l
    best_obj = obj

        
    # N回の2-opt
    is_good_swap = True
    for _ in tqdm(range(4000)):

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
            sub_solution = solution[target_nodes[0]:target_nodes[1]+1]
            
            # reverse order
            tmp_solution[target_nodes[0]:target_nodes[1]+1] = sub_solution[::-1]

            # calculate the length of the tour
            obj = length(points[tmp_solution[-1]], points[tmp_solution[0]])
            for index in range(0, nodeCount-1):
                obj += length(points[tmp_solution[index]], points[tmp_solution[index+1]])

            if obj < best_obj:
                best_obj = obj
                solution = copy.copy(tmp_solution)
                print("solution is updated!", obj)
                is_good_swap = True
                
                if viz:
                    plt.scatter([e.x for e in points], [e.y for e in points], s=10)
                    for i in range(len(solution)):
                        n1 = solution[i]
                        n2 = solution[(i+1)%len(solution)]
                        plt.plot([points[n1].x, points[n2].x], [points[n1].y, points[n2].y], "-", c="gray" )                    
                        
                    plt.show()
                
                break

    # １回スワップするのに１番効果のあるものを探して実行する(おそそう)
    # is_good_swap = True
    # for _ in tqdm(range(4000)):

    #     if not is_good_swap:
    #         print("No more good swap")
    #         break

    #     # random sampling
    #     # target_nodes = random.sample(node_list, 2)
    #     is_good_swap = False
    #     # 一番いいswapをさがす　
    #     obj = length(points[solution[-1]], points[solution[0]])
    #     for index in range(0, nodeCount-1):
    #         l = length(points[solution[index]], points[solution[index+1]])
    #         obj += l
    #     best_obj = obj

    #     best_solution = None
    #     for target_nodes in it.combinations(solution, 2):
            
    #         idx1 = solution.index(target_nodes[0])
    #         idx2 = solution.index(target_nodes[1])
    #         if np.fabs(idx1 - idx2) == 1:
    #             continue
            
    #         if idx1 == 0 and idx2 == len(solution)-1: # 末尾
    #             continue
            
    #         # print(target_nodes)
    #         tmp_solution = copy.copy(solution)    
            
    #         # 2-OPT 
    #         sub_solution = solution[target_nodes[0]:target_nodes[1]+1]
            
    #         # reverse order
    #         tmp_solution[target_nodes[0]:target_nodes[1]+1] = sub_solution[::-1]

    #         # calculate the length of the tour
    #         obj = length(points[tmp_solution[-1]], points[tmp_solution[0]])
    #         for index in range(0, nodeCount-1):
    #             obj += length(points[tmp_solution[index]], points[tmp_solution[index+1]])

    #         if obj < best_obj:
    #             best_obj = obj
    #             best_nodes = target_nodes # 最後にこれをつかう
    #             best_solution = copy.copy(tmp_solution)
    #             # print("solution is updated!", obj)
    #             is_good_swap = True
                
    #     if is_good_swap:
    #         solution = copy.copy(best_solution)
    #         if viz:
    #             print("solution is updated!", obj)
    #             plt.scatter([e.x for e in points], [e.y for e in points], s=10)
    #             for i in range(len(solution)):
    #                 n1 = solution[i]
    #                 n2 = solution[(i+1)%len(solution)]
    #                 plt.plot([points[n1].x, points[n2].x], [points[n1].y, points[n2].y], "-", c="gray" )                    
                    
    #             plt.show()
                

    plt.scatter([e.x for e in points], [e.y for e in points], s=10)
    for i in range(len(solution)):
        n1 = solution[i]
        n2 = solution[(i+1)%len(solution)]
        plt.plot([points[n1].x, points[n2].x], [points[n1].y, points[n2].y], "-", c="gray" )                    
        
    plt.show()



    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data

def solve_it_tabu(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    solution = [] # range(0, nodeCount)

    # Nearest Neighborで探索する
    points_buffer = copy.copy(points)

    sol = 0
    solution.append(sol)
    node_list = list(range(0, nodeCount))
    node_list.pop(sol)

    # NNで初期値作成
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
            
        if len(node_list)==0:
            break

    # 初期値
    # solution = list(range(0, nodeCount))


    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        l = length(points[solution[index]], points[solution[index+1]])
        obj += l
    best_obj = obj
        

    # １回スワップするのに１番効果のあるものを探して実行する(おそそう)
    is_good_swap = True
    for _ in tqdm(range(4000)):

        if not is_good_swap:
            print("No more good swap")
            break

        # random sampling
        # target_nodes = random.sample(node_list, 2)
        is_good_swap = False
        # 一番いいswapをさがす　
        obj = length(points[solution[-1]], points[solution[0]])
        for index in range(0, nodeCount-1):
            l = length(points[solution[index]], points[solution[index+1]])
            obj += l
        best_obj = obj

        best_solution = None
        for target_nodes in it.combinations(solution, 2):
            
            idx1 = solution.index(target_nodes[0])
            idx2 = solution.index(target_nodes[1])
            if np.fabs(idx1 - idx2) == 1:
                continue
            
            if idx1 == 0 and idx2 == len(solution)-1: # 末尾
                continue
            
            # print(target_nodes)
            tmp_solution = copy.copy(solution)    
            
            # 2-OPT 
            sub_solution = solution[target_nodes[0]:target_nodes[1]+1]
            
            # reverse order
            tmp_solution[target_nodes[0]:target_nodes[1]+1] = sub_solution[::-1]

            # calculate the length of the tour
            obj = length(points[tmp_solution[-1]], points[tmp_solution[0]])
            for index in range(0, nodeCount-1):
                obj += length(points[tmp_solution[index]], points[tmp_solution[index+1]])

            if obj < best_obj:
                best_obj = obj
                best_nodes = target_nodes # 最後にこれをつかう
                best_solution = copy.copy(tmp_solution)
                # print("solution is updated!", obj)
                is_good_swap = True
                
        if is_good_swap:
            solution = copy.copy(best_solution)
            if viz:
                print("solution is updated!", obj)
                plt.scatter([e.x for e in points], [e.y for e in points], s=10)
                for i in range(len(solution)):
                    n1 = solution[i]
                    n2 = solution[(i+1)%len(solution)]
                    plt.plot([points[n1].x, points[n2].x], [points[n1].y, points[n2].y], "-", c="gray" )                    
                    
                plt.show()
                

    plt.scatter([e.x for e in points], [e.y for e in points], s=10)
    for i in range(len(solution)):
        n1 = solution[i]
        n2 = solution[(i+1)%len(solution)]
        plt.plot([points[n1].x, points[n2].x], [points[n1].y, points[n2].y], "-", c="gray" )                    
        
    plt.show()



    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

