#!/usr/bin/env python3

from collections import Counter

import numpy as np


class Impurity:

    def __init__(self, data):
        self.set_cardinality(data)

    def set_cardinality(self, data):
        self.n = len(data)
        self.count = Counter(data)


class GiniImpurity(Impurity):

    def __init__(self, data):
        super().__init__(data)

    def get_impurity(self):
        return 1 - sum((value / self.n)**2 for _, value in self.count.items())


class CrossEntropyImpurity(Impurity):

    def __init__(self, data):
        super().__init__(data)

    def get_impurity(self):
        return sum(-(value/self.n)*np.log(value/self.n) for _, value in self.count.items())


if __name__ == "__main__":

    gi = GiniImpurity([1, 2, 2])
    print(gi.get_impurity())

    gi = CrossEntropyImpurity([1, 2, 2])
    print(gi.get_impurity())