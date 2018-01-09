# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 10:55:45 2017

@author: eduai_000
"""

import numpy as np
import matplotlib.pyplot as plt


deepDel = 16
delta = [(0.1 ** i) for i in range(deepDel)]

x0 = 1.5
trueVal = np.cos(x0)

dfCentre = [(np.sin(x0 + d) - np.sin(x0 - d)) / (2 * d) for d in delta] - trueVal
dfAval = [(np.sin(x0 + d) - np.sin(x0)) / d for d in delta] - trueVal

logCentre = np.log10(np.abs(dfCentre))
logAval = np.log10(np.abs(dfAval))

dfComplex = [np.imag(np.sin(x0 + d * 1j)) / d for d in delta] - trueVal
logComplex = np.log10(np.abs(dfComplex))


plt.plot(logCentre, 'rs', label="DF centrées")
plt.plot(logAval, 'b.', label="DF aval")
plt.plot(logComplex, 'g^', label="DF complexes")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.ylabel('log(erreur relative)')
plt.xlabel('-log(delta)')
plt.show()
