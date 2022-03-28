from typing import Dict, Tuple
import numpy as np
import random
random.seed(0)


def gen_env(price, prob=0.5):
    """
    基準となる価格の2倍か1/2倍のお金のペアを生成
    probが1/2倍のお金を生成する確率
    """
    r = random.uniform(0, 1)
    if r > prob:
        pair_price = price * 2.0
    else:
        pair_price = price / 2.0
    return price, pair_price


def get_money(env, prob=0.5):
    r = random.uniform(0, 1)
    if r > prob:
        i = 0
    else:
        i = 1
    return env[i], i

"""
a'= 10000

a'= aのとき　a= 10000 --> 1/2a or 2a --> 2.5/2 = 1.25

a'=10000
a' = 2aのとき   a = 5000  --> a or 2a   --> 1.5a = 7500
a' = 1/2aのとき a = 20000 --> a or 1/2a --> 0.75a = 15000


(1.25 + 1.25 + 1.5 + 0.75) = 

[(1/2a, a), (a, 2a)]
"""

N = 100000  # 回
swap_count = 0  # 回
root_price = 10000  # 円
total = 0  # 円
for _ in range(N):

    # 封筒を作成
    envelope = gen_env(root_price, prob=0.5) # root, pairの順

    # お金を選択
    money, idx = get_money(envelope, prob=0.5)

    # 中身をみて1万円だったら交換する
    # if money == root_price:
    if money != root_price:

        swap_count += 1
        if idx == 0:
            total += envelope[1]  # シミュレーションごとに加算
        else:
            total += envelope[0]

# 平均
average = total/swap_count
print(f"{average=}")
print(f"{swap_count/N=}")

# ==============================
# average=12501.053096102542
# ratio of swap: 0.49853
