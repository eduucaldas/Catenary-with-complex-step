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

# if you want to change paramenters you can do it when creating the catenary, the default values are: N=50, alfa=0.1 ** 3, delta=0.1 ** 10, eps=0.1**3. You can change many other things as well just look at the __init__ method
# This is how you change the parameters in any of these functions
# cat = catenary(eps=eps, N=N, L=L, alfa=alfa, delta=delta, a=a, b=b)
# or just change in the standards above


# maxIt so you don`t wait forever for the solution
std_maxIt = 100000


# You shouldn't need to read the class implementation
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
        std_chain0 = [-0.1 for i in range(self.N)]
        std_chain0[0] = self.a
        std_chain0[self.N - 1] = self.b
        self.solutions = [std_chain0]
        self.maxIt = std_maxIt / self.N  # we dont want to wait forever
        self.solAnalytic = [math.cosh(j * (1. / (self.N - 1)) - 0.5) - math.cosh(0.5) for j in range(self.N)]

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

    def gradJ(self, chain):
        D = [0] + [self.delJ(chain, i) for i in range(1, len(chain) - 1)] + [0]  # the borders are fixed
        return D

    def evolution(self):
        D = self.gradJ(self.solutions[-1])

        self.solutions.append([s - d * (self.alfa / np.linalg.norm(D, 2)) for s, d in zip(self.solutions[-1], D)])

    def plot(self, k=-1):
        # plots a catenary that is in the vector of solutions
        rangeX = [i * self.h for i in range(self.N)]
        plt.plot(rangeX, self.solutions[k], 'r-', label="optimal")

        if(self.a == 0 and self.b == 0):
            plt.plot(rangeX, self.solAnalytic, 'r--', label="analytic")

        if(k == -1 or k == len(self.solutions) - 1):
            plt.title("Optimal Solution #{}".format(len(self.solutions) - 1))
        else:
            plt.title("intermediate Solution #{}".format(k))
        plt.show()

    def display_J(self):
        rangeX = range(len(self.solutions))
        plt.plot(rangeX, [self.J(s) for s in self.solutions], 'b-')
        plt.title("Graph of J")
        plt.ylabel('J')
        plt.xlabel('iteration')
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

    def stopConditionGrad(self):
        # stopping Condition: gradJ close to zero. This will work if the function is k lipschitzian, which it is. But i couldnt estimate k to do it. Not used
        if(np.linalg.norm(self.gradJ(self.solutions[-1]), 2) < 20 * self.alfa * (0.1 + self.alfa / self.eps) or len(self.solutions) >= self.maxIt):
            return True
        else:
            return False

    def stopConditionDiff(self):
        # stopping Condition: difference between energies calculated
        if(len(self.solutions) > 1 and np.abs(self.J(self.solutions[-1]) - self.J(self.solutions[-2])) < (0.1 ** 11) * (1 / self.eps) or len(self.solutions) >= self.maxIt):
            return True
        else:
            return False

    def stopConditionCloseAnalytical(self):
        # stopping Condition: comparing with the analytical solution for the base case, used for testing
        if(np.linalg.norm(np.array(self.solutions[-1]) - np.array(self.solAnalytic), 2) < math.sqrt(self.N) * self.alfa or len(self.solutions) >= self.maxIt):
            return True
        else:
            return False

    def evolve(self):
        # evolving until finds stopCondition
        while not self.stopConditionDiff():  # change here the stopping condition
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


def test_delJ(cat):
    print([cat.delJ(cat.solutions[-1], i) for i in range(cat.N)])


def test_gradJ(cat):
    print(cat.gradJ(cat.solutions[-1]))


def test_evolution(cat):
    # this basically the same as asked in 7, used mainly for testing
    cat.evolve()
    print(cat.eps)
    # 10 intermediate solutions
    for i in range(0, len(cat.solutions), math.ceil(len(cat.solutions) / 10.)):
        cat.plot(i)
        print("J = ", np.linalg.norm(cat.J(cat.solutions[i])), "\n")
    # final solution
    cat.plot()

    print("Solutions Computed: ", len(cat.solutions), "\noptimized J: ", cat.J(cat.solutions[-1]))
    if(cat.a == 0 and cat.b == 0):
        print("J with analytical solution: {}".format(cat.J(cat.solAnalytic)))
    cat.display_J()
    cat.display_energies()


def question6():
    # not sure what she meant with real energy
    cat = catenary()
    print("Energy = ", cat.energy(cat.solutions[0]))
    print("J = ", cat.J(cat.solutions[0]))


def question7():
    # plots the evolution of the solution, as well as intermediary solutions and the evolution of energies
    cat = catenary()
    cat.evolve()
    # 10 intermediate solutions
    for i in range(0, len(cat.solutions), math.ceil(len(cat.solutions) / 10.)):
        cat.plot(i)
        print(np.linalg.norm(cat.gradJ(cat.solutions[i])))
    # final solution
    cat.plot()

    cat.display_J()
    cat.display_energies()


def question9():
    # plots the evolution of the energies according to each epsilon
    nEps = 4    # number of eps youll test
    logEpsInit = 2
    cat = [catenary(0.1 ** i) for i in range(logEpsInit, logEpsInit + nEps)]
    for c in cat:
        print("\n\nesp = {:.2E}".format(c.eps))
        c.evolve()
        c.display_energies()


def question10():
    # Compares the two methods' solutions with the analytical solution for different values for delta
    nDelta = 6  # how many times we`ll change delta
    dBegin = 2  # it takes a while to see the difference, you may want to set this to 2
    N = 50
    deltas = [std_delta * (0.1 ** i) for i in range(dBegin, nDelta)]
    catComplex = [catenary(delta=d) for d in deltas]
    catCentral = [catenary2(delta=d) for d in deltas]
    solAnalytic = [math.cosh(j * (1. / (N - 1)) - 0.5) - math.cosh(0.5) for j in range(N)]
    rangeX = [i * (1. / (N - 1)) for i in range(N)]
    difCentral = []
    difComplex = []

    for i in range(nDelta - dBegin):
        catComplex[i].evolve()
        catCentral[i].evolve()
        plt.title("Solutions with delta = {:.1E}".format(deltas[i]))
        plt.plot(rangeX, catComplex[i].solutions[-1], 'r-', label='complex')
        plt.plot(rangeX, catCentral[i].solutions[-1], 'b-', label='central')
        plt.plot(rangeX, solAnalytic, 'g--', label='analytical')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc="best",
                   ncol=2, mode="expand", borderaxespad=0.)
        plt.show()

        difComplex.append(np.linalg.norm(np.array(catComplex[i].solutions[-1]) - np.array(solAnalytic), 2))
        difCentral.append(np.linalg.norm(np.array(catCentral[i].solutions[-1]) - np.array(solAnalytic), 2))

    rangeX = -np.log10(deltas)
    plt.plot(rangeX, difCentral, 'b-', label="central")
    plt.plot(rangeX, difComplex, 'r-', label="complex")
    plt.show()



# question6()
# question7()
# question9()
question10()

# these are for testing, catenary2 is the one with the real method
# cat = catenary()
# cat = catenary2()
# test_delJ(cat)
# test_gradJ(cat)
# test_evolution(cat)
