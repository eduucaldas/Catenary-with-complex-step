# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 10:55:45 2017

@author: eduai_000
"""

# Resources
from mpmath import *
import matplotlib.pyplot as plt
# mpmath is used for bigger precision
# To install mpmath type "pip install mpmath" on the terminal

# Sets number of decimal places
mp.dps = 33
mp.pretty = True

deepDel = 16
delta = [power(mpf('0.1'), i) for i in range(deepDel)]

x0 = mpf('1.5')
trueVal = cos(x0)

dfCentre = [fsub(fdiv(fsub(sin(fadd(x0, d)), sin(fsub(x0, d))), (2 * d)), trueVal) for d in delta]
dfAval = [fsub(fdiv(fsub(sin(fadd(x0, d)), sin(x0)), d), trueVal) for d in delta]

logCentre = [float(log(abs(d), 10)) for d in dfCentre]
logAval = [float(log(abs(d), 10)) for d in dfAval]

dfComplex = [fsub(fdiv(im(sin(mpc(x0, d))), d), mpf(trueVal)) for d in delta]
logComplex = [float(log(d, 10)) for d in dfComplex]

plt.plot(logCentre, 'rs', label="DF centrées")
plt.plot(logAval, 'b.', label="DF aval")
plt.plot(logComplex, 'g^', label="DF complexes")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.ylabel('log(erreur relative)')
plt.xlabel('-log(delta)')
plt.show()
