

# to install mpmath do
# Follow instructions in http://mpmath.org/doc/current/setup.html#download-and-installation
# 1) download from https://pypi.python.org/packages/7a/05/b3d1472885d8dc0606936ea5da0ccb1b4785682e78ab15e34ada24aea8d5/mpmath-1.0.0.tar.gz
# 2) extraxt the contents from the downloaded package
# 3) Open terminal and navigate to the directory with the contents
# 3)    tap python setup.py install
from mpmath import *
import matplotlib.pyplot as plt

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

# this should work
plt.plot(range(deepDel), logCentre, 'r--', logAval, 'bs', logComplex, 'g^')
plt.show()
