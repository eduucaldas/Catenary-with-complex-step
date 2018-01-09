import numpy as np
import matplotlib.pyplot as plt
import cmath as cm
import math

# standard parameters:
std_eps = 0.001
std_N = 50
std_L = 2 * math.sinh(0.5)
std_alfa = 0.001
std_delta = 0.1 ** 10
std_a = 0
std_b = 0
std_chain0 = [-0.1 for i in range(std_N)]
std_chain0[0] = std_a
std_chain0[std_N - 1] = std_b


class catenary:
    def __init__(self, eps=std_eps, N=std_N, L=std_L, alfa=std_alfa, delta=std_delta, a=std_a, b=std_b):
        self.N = N
        self.h = 1. / (N - 1)
        self.eps = eps
        self.L = L
        self.alfa = alfa
        self.delta = delta
        self.a = a
        self.b = b
        self.solutions = [std_chain0]

    def energy(self, chain):
        v = np.array(chain[:- 1])
        v_plus = np.array(chain[1:])
        return self.h * np.sum([((j + j_plus) / 2.) * cm.sqrt(1 + (((j_plus - j) / self.h) * ((j_plus - j) / self.h))) for j_plus, j in zip(v_plus, v)])

    def constraint_term(self, chain):
        v = chain[:-1]
        v_plus = chain[1:]
        return ((self.h * np.sum([cm.sqrt(1 + (((j_plus - j) / self.h) ** 2)) for j_plus, j in zip(v_plus, v)]) - self.L) ** 2) / self.eps

    def J(self, chain):
        return self.energy(chain) + self.constraint_term(chain)

    def delJ(self, chain, i):
        # implemented with complex step method
        c = chain[:]
        c[i] += self.delta * 1j
        Del = np.imag(self.J(c)) / self.delta
        return float(Del)

    def divJ(self, chain):
        D = [0]  # fixed ends

        for i in range(1, len(chain) - 1):
            D.append(self.delJ(chain, i))

        D.append(0)  # fixed ends
        return D

    def evolution(self):
        D = self.divJ(self.solutions[-1])

        self.solutions.append([s - d * (self.alfa / np.linalg.norm(D, 2)) for s, d in zip(self.solutions[-1], D)])

    def plot(self, k=-1):
        # plots a catenary that is in the vector of solutions
        rangeX = [i * self.h for i in range(self.N)]
        plt.plot(rangeX, self.solutions[k], 'r-')
        if(k == -1 or k == self.N):
            plt.title("Optimal Solution")
        else:
            plt.title("intermediate Solution #{}".format(k))
        plt.show()

    def display_J(self):
        rangeX = range(len(self.solutions))
        plt.plot(rangeX, [self.J(s) for s in self.solutions], 'b-')
        plt.title("Graph of J")
        plt.show()

    def display_energies(self):
        # Displays the true energy and the constraint energy separately
        rangeX = range(len(self.solutions))
        plt.plot(rangeX, [self.energy(s) for s in self.solutions], 'b-', label="energy")
        plt.plot(rangeX, [self.constraint_term(s) for s in self.solutions], 'r-', label="constraint")
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
        plt.title("Graph of the Energies")
        plt.ylabel('energies')
        plt.xlabel('iteration')
        plt.show()

    def stopCondition(self):
        # stopping Condition: divJ close to zero but how much? this 0.3 constant comes from where?
        if(np.linalg.norm(self.divJ(self.solutions[-1]), 2) < 0.03 * self.alfa * (1. / self.eps) or len(self.solutions) * self.N > 50000):
            return True
        else:
            return False

    def evolve(self):
        # evolving until finds stopCondition
        while not self.stopCondition():
            self.evolution()


class catenary2(catenary):
    # this is an overloading to test solving the problem with central finite differences method for the derivative
    def delJ(self, chain, i):
        c1 = np.array(chain[:])
        c2 = np.array(chain[:])
        c1[i] -= self.delta
        c2[i] += self.delta
        Del = (self.J(c2) - self.J(c1)) / (2 * self.delta)
        return float(Del)


def question6():
    # not sure what she meant with real energy
    cat = catenary()
    print("Energy = ", cat.energy(cat.solutions[0]))
    print("J = ", cat.J(cat.solutions[0]))


def test_delJ(cat):
    print([cat.delJ(cat.solutions[0], i) for i in range(cat.N)])


def test_divJ(cat):
    print(cat.divJ(cat.solutions[0]))


def test_evolution(cat):
    cat.evolve()
    # 10 intermediate solutions
    for i in range(0, len(cat.solutions), math.ceil(len(cat.solutions) / 10.)):
        cat.plot(i)
        print("J = ", np.linalg.norm(cat.J(cat.solutions[i])), "\n")
    # final solution
    cat.plot()

    print("Solutions Computed: ", len(cat.solutions), "\noptimized J: ", cat.J(cat.solutions[-1]))
    cat.display_J()
    cat.display_energies()


def question7():
    # plots the evolution of the solution, as well as intermediary solutions and the evolution of energies
    cat = catenary()
    cat.evolve()
    # 10 intermediate solutions
    for i in range(0, len(cat.solutions), math.ceil(len(cat.solutions) / 10.)):
        cat.plot(i)
        print(np.linalg.norm(cat.divJ(cat.solutions[i])))
    # final solution
    cat.plot()

    cat.display_J()
    cat.display_energies()


def question9():
    # plots the evolution of the energies according to each epsilon
    nEps = 5
    logEpsInit = 2
    cat = [catenary(0.1 ** i) for i in range(logEpsInit, logEpsInit + nEps)]
    for c in cat:
        c.evolve()
        c.display_energies()


def question10():
    # Compares the two methods' solutions with the analytical solution
    nDelta = 5  # how many times we`ll cahnge delta
    logDeltaInit = 10
    N = 50

    catComplex = [catenary(delta=0.1 ** i) for i in range(logDeltaInit, logDeltaInit + nDelta)]
    catCentral = [catenary2(delta=0.1 ** i) for i in range(logDeltaInit, logDeltaInit + nDelta)]
    solAnalytic = [math.cosh(j * (1. / (N - 1)) - 0.5) - math.cosh(0.5) for j in range(N)]
    rangeX = [i * (1. / (N - 1)) for i in range(N)]

    for i in range(nDelta):
        catComplex[i].evolve()
        catCentral[i].evolve()
        plt.plot(rangeX, catComplex[i].solutions[-1], 'r-', label='complex')
        plt.plot(rangeX, catCentral[i].solutions[-1], 'b-', label='central')
        plt.plot(rangeX, solAnalytic, 'g--', label='analytical')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
        plt.show()


# question6()
# question7()
# question9()
# question10()

# these are for testing, catenary2 is the one with the real method
# cat = catenary()
# cat = catenary2()
# test_delJ(cat)
# test_divJ(cat)
# test_evolution(cat)
