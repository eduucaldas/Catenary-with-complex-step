import numpy as np
import matplotlib.pyplot as plt
from cmath import *

# Initial parameters, gotta probs create a class


class catenary:

    def __init__(self, N=50, eps=0.001, L=sinh(0.5), alfa=0.001, delta=0.1 ** 10, a=0, b=0):
        self.N = N
        self.h = 1. / (N - 1)
        self.eps = eps
        self.L = L
        self.alfa = alfa
        self.delta = delta
        self.a = a
        self.b = b
        chain0 = [-0.1 for i in range(N)]
        chain0[0] = a
        chain0[N - 1] = b
        self.solutions = [chain0]

    # we gotta have chain with the right length i.e. N
    def energy(self, chain):
        v = chain[:- 1]
        v_plus = chain[1:]
        return self.h * np.sum([((j + j_plus) / 2.) * sqrt(1 + (((j_plus - j) / self.h) ** 2)) for j_plus, j in zip(v_plus, v)])

    def constraint_term(self, chain):
        v = chain[:- 1]
        v_plus = chain[1:]
        return ((self.h * np.sum([sqrt(1 + (((j_plus - j) / self.h) ** 2)) for j_plus, j in zip(v_plus, v)]) - self.L) ** 2) / self.eps

    def J(self, chain):
        return self.energy(chain) + self.constraint_term(chain)

    def delJ(self, chain, i):
        c = chain[:]
        c[i] += self.delta * 1j
        Del = np.imag(self.J(c)) / self.delta
        return Del

    def divJ(self, chain):
        D = [0]

        for i in range(1, len(chain) - 1):
            D.append(self.delJ(chain, i))
        D.append(0)
        return D

    def evolution(self):
        D = self.divJ(self.solutions[-1])

        self.solutions.append([s - d * (self.alfa / np.linalg.norm(D, 2)) for s, d in zip(self.solutions[-1], D)])

    def plot(self):
        rangeX = [i * self.h for i in range(self.N)]
        plt.plot(rangeX, self.solutions[-1], 'r-')
        plt.show()

    def isCloseEnough(self):
        if(np.linalg.norm(self.divJ(self.solutions[-1]), 2) < 0.00001 and len(self.solutions) * self.N > 1000):
            return True
        else:
            return False


def question6():
    # not sure what she meant as real energy
    cat = catenary()
    print("Energy = ", cat.energy(cat.solutions[0]))
    print("J = ", cat.J(cat.solutions[0]))


def test_delJ():
    cat = catenary()
    print([cat.delJ(cat.solutions[0], i) for i in range(cat.N)])


def test_divJ():
    cat = catenary()
    print(cat.divJ(cat.solutions[0]))


def test_evolution():
    cat = catenary()
    count = 0
    while not cat.isCloseEnough():
        cat.evolution()
        if(len(cat.solutions) % 100 == 0):
            cat.plot()
    cat.plot()



# question6()
# test_delJ()
# test_divJ()
test_evolution()
