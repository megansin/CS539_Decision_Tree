import numpy as np
from sklearn import tree
from part1 import *

# task 2-1
df = np.loadtxt("credit.txt", dtype=object)

X = df[1:, 1:-1].T
Y = df[1:, -1]

t = Tree.train(X, Y)

'''
input: t (tree node)
look at attributes in C and which attribute (i.e. debt)
if not a leaf node -> recursion input each children (nodes)
else print prediction

This is what I want it to print out (end goal):
Income = low
| Married? = no
    | Debt = low: low
    | Debt = medium
        | Gender = male: high
        | Gender = female: low
| Married? = yes: high
Income = medium: low
Income = high: low
'''

def print_tree(t, counter=0):
    line = '  '
    if t.isleaf is False:
        for attribute in t.C:
            print(counter*line + "|",  df[0, t.i+1], "=", attribute)
            print_tree(t.C[attribute], counter+1)
    else:
        # TODO: figure out how to print the prediction on the previous line
        print(": "+ t.p)

print_tree(t)

Tom = ["low", "low", "no", "yes", "male"]
Ana = ["low", "medium", "yes", "yes", "female"]
X_2 = np.array([Tom, Ana]).T
Y_2 = Tree.predict(t, X_2)
ans = {}
ans["Tom"] = Y_2[0]
ans["Ana"] = Y_2[1]
print("Predictions", ans)

# task 2-2
Y_edited = Y
Y_edited[8] = "high"

t_2 = Tree.train(X, Y_edited)
print_tree(t_2)
# TODO: also list/describe how the trees are different...

# Looking at both decision trees, owning property did not play a role in either decision trees. 
# The original decision tree utilized all of the other attributes. 
# In the new decision tree, gender no longer plays a role like it did in the original decision tree.