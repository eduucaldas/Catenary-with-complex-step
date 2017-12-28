import numpy as np
import matplotlib.pyplot as plt
import scipy as sps

#to install mpmath do
#Follow instructions in http://mpmath.org/doc/current/setup.html#download-and-installation
#1) download from https://pypi.python.org/packages/7a/05/b3d1472885d8dc0606936ea5da0ccb1b4785682e78ab15e34ada24aea8d5/mpmath-1.0.0.tar.gz
#2) extraxt the contents from the downloaded package
#3) Open terminal and navigate to the directory with the contents
#3)	tap python setup.py install
from mpmath import *


deepDel = 16
delta = [(0.1 ** i) for i in range(deepDel)]

x0 = 1.5
trueVal = np.cos(x0)

dfCentre = [(np.sin(x0+d) - np.sin(x0-d))/(2*d) for d in delta] - trueVal
dfAval = [(np.sin(x0+d) - np.sin(x0))/d for d in delta] - trueVal 

logCentre = np.log10(np.abs(dfCentre))
logAval = np.log10(np.abs(dfAval))

mp.dps = 33
mp.pretty = True
x0 = mpf('1.5')
trueVal = cos(x0)
dfComplex = [fsub( fdiv( im( sin( mpc(x0, d) ) ), d ), mpf(trueVal)) for d in delta]
logComplex = [float(log(d, 10)) for d in dfComplex]

#this should work
plt.plot(logCentre, logAval, logComplex, range(deepDel))