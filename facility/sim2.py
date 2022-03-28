import random
# random.seed(0)

swap_total = 0
normal_total = 0
if __name__ == '__main__':

    N= 10000
    x = 10000 # yen
    total = 0
    for _ in range(N):

        x = 10000  # yen
        r = random.randint(0, 1)
        # 封筒作成

        if r == 0:
            e = (x, 2*x)
        else:
            e = (x, x/2)

        if r == 0:
            e = (x, x/2)
        else:
            e = (x/2, x/4)

        if r == 0:
            e = (2*x, x)
        else:
            e = (2*x, 4*x)

        # normal
        idx = random.randint(0, 1)
        normal_total += e[idx]

        # swap
        idx = random.randint(0, 1)
        while e[idx] != x:
            idx = random.randint(0, 1)

        if idx == 0:
            swap_total += e[1]
        else:
            swap_total += e[0]


print(f"{swap_total/N=}")
print(f"{normal_total/N=}")