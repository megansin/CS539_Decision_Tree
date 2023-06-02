import numpy as np
from sklearn import tree
from part1 import *

df = np.loadtxt("credit.txt", dtype=object)

X = df[1:, 1:-1].T
Y = df[1:, -1]

t = Tree.train(X, Y)
print(t.C.keys())

print(df[0, t.i])

for i in t.C:

    print(i)

print(t.i)
print(t.C)
print(t.isleaf)

