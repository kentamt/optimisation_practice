#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import random
import copy
import numpy as np
import itertools as it
from tqdm import tqdm
random.seed(20200627)
sys.setrecursionlimit(2000)
print(sys.getrecursionlimit())

class Vertex:
    def __init__(self, name : int, n_colors : int):
        self.name = name
        self.is_visited = False   
        self.neighbors = [] # list of vertex
        self.available_colors = [True] * n_colors # 初期値はすべての色が使える
        self.n_neighbors = 0
    
    def __str__(self):
        return f"{self.name}"
    
    def clear(self):
        self.is_visited = False
        self.available_colors = [True] * len(self.available_colors)
    
    def visit(self):
        self.is_visited = True
        
    def add_neighbors(self, v):
        self.neighbors.append(v)
        self.n_neighbors = len(self.neighbors)

class Graph:
    def __init__(self):
        self.vertex_dict = {}
    
    def clear(self):
        for v in self.vertex_dict.values():
            v.clear()
    
    def add_node(self, v : Vertex):
        # print(f"try to add {v}")

        if len(self.vertex_dict) == 0:
            # print(f"add {v}")
            self.vertex_dict[v.name] = v
            return 1
        
        is_same_node = False
        for e in self.vertex_dict.keys():
            if v.name == e:
                is_same_node = True
                break

        if not is_same_node:
            # print(f"add {v}")
            self.vertex_dict[v.name] = v
        else:
            # print(f"warn: {v} is already registored.")
            pass
        
    def add_edge(self, v1 : Vertex, v2 : Vertex):
        """ オブジェクトを保つために慎重に作業している """
        
        _v1 = self.vertex_dict[v1.name]
        _v2 = self.vertex_dict[v2.name]        
        _v1.add_neighbors(_v2)
        _v2.add_neighbors(_v1)
        
        

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))
    

    # 無向グラフの作成
    G = Graph()
    for e in edges:
        # print(e)        
        fr = Vertex(e[0], node_count)
        to = Vertex(e[1], node_count)
        G.add_node(fr)
        G.add_node(to)
        G.add_edge(fr, to) # 両方のエッジを張るので2回やらなくてOK
                
                
    # print("neighbors")
    # for node in G.vertex_dict.values():
    #     for neighbor in node.neighbors:
    #         print(f"{node.name}-->{neighbor.name}")
    #         pass


    # 以下、色塗り
    # print("loop count: {loop_count}")
    G.clear()

    # random_node_name = random.sample(list(range(0, node_count)), node_count)    
    sorted_nodes = sorted([e for e in G.vertex_dict.values()], key=lambda x:x.n_neighbors, reverse=True)
                    
    colors = range(0, node_count) # 色番号
    cid = 0 # 今最大の色番号
    solution = [] # (node_id, color)
    visited = []

    for cv in sorted_nodes:
        # あたらしいノードに行くたびに毎回制約条件の伝搬を行う
        for node, color in solution:
            G.vertex_dict[node.name].available_colors[color] = False
            for neighbor in node.neighbors:
                neighbor.available_colors[color] = False
        
        if not cv.is_visited:
            # 訪問していないのでドメインから色を決める
            
            # 色を決める
            for ic, c in enumerate(colors): # 色. 1~node_count
                # print(cv.name, cv.available_colors)            
                if cv.available_colors[ic] == True: # 前から順番に使えるものを使う
                    # 使える場合はcを使う
                    solution.append((cv, c)) # node_id, color
                    # print(f"{cv.name}, color={c}")
                    break
            
            cv.visit()
            visited.append(cv)
                
        else:
            pass # 訪問済みのためなにもしない
        
    # print("solutin:")
    # for n, c in solution:

    # build a trivial solution
    solution = sorted(solution, key=lambda x:x[0].name)
    solution = [e[1] for e in solution]            
    value = max(solution) + 1
    opt = 1
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(opt) + '\n'
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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

