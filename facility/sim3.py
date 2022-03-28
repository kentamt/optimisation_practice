import random
# random.seed(0)

if __name__ == '__main__':

    N= 100000
    total = 0
    p = 0
    d = 0

    for _ in range(N):

        a = random.randint(1, 5000)
        b = a * 2
        arr = [a, b]
        random.shuffle(arr)
        # print(arr[0])
        # print(arr[1])
        if arr[0] > arr[1]:
            p += 1
        d += arr[1] - arr[0]

print(f"{p=}")
print(f"{d=}")
print(f"{d/N=}")