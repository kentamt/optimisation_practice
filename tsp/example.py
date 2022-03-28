import itertools as it
import numpy as np

l = list(range(0, 10))

for c in it.combinations(l, 2):
    idx1 = l.index(c[0])
    idx2 = l.index(c[1])
    if np.fabs(idx1-idx2) == 1:
        continue
    print(c)