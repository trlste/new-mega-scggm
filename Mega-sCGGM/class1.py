#!/bin/python

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ssp
import scipy.sparse.linalg as ssl
import sys

sys.path.append("../Mega-sCGGM/")
from mega_scggm import mega_scggm
from two_mega_scggm import two_mega_scggm

n = 100
p = 50-100
q = 100-150
r = 10
#diagonal matrix
#qxq, V
V = ssp.eye(q, format="coo")
V.setdiag(0.3, 1)
V.setdiag(0.3, -1)
#diagonal matrix
#qxq, Gamma
Gamma = ssp.eye(q, format="coo")
Gamma.setdiag(0.3, 1)
Gamma.setdiag(0.3, -1)

#p*q check 
p_influence = int(math.floor(p * 0.5))
Psi_top = ssp.random(p_influence, q)
Psi_bottom = ssp.coo_matrix((p-p_influence,q))
Psi = ssp.vstack([Psi_top, Psi_bottom])

#(p+r)*q
p_influence = int(math.floor(p * 0.5))
F_top = ssp.random(p_influence, q)
F_bottom = ssp.coo_matrix((p-p_influence,q))
r=ssp.random(r,q)
F = ssp.vstack([F_top, F_bottom,r])

X = np.random.binomial(n=1, p=0.5, size=(n,2*p+r))
X_r_dropped=X[:,:2*p].reshape(2*n,-1)
X_r_dropped_p=X_r_dropped[::2,:]
X_r_dropped_m=X_r_dropped[1::2,:]
X_sum=np.concatenate((X_r_dropped_m+X_r_dropped_p,X_r),axis=1)
X_diff=np.subtract(X_r_dropped_p, X_r_dropped_m)

Sigma_V = ssl.inv(V)
Sigma_Gamma=ssl.inv(Gamma)

meanSum = -1 * X_sum @ F @ Sigma_V
diffSum= -1 * X_diff @ Psi @ Sigma_Gamma

try:
    import sksparse.cholmod as skc
    V_factor = skc.cholesky(V)
    Gamma_factor=skc.cholesky(Gamma)
    noiseSum = (V_factor.solve_Lt(np.random.randn(q,n))).transpose()
    noiseDiff = (Gamma_factor.solve_Lt(np.random.randn(q,n))).transpose()
except:
    noiseSum = np.random.multivariate_normal(np.zeros(q), Sigma_V.todense(), size=n)
    noiseDiff = np.random.multivariate_normal(np.zeros(q), Sigma_Gamma.todense(), size=n)

Ysum = meanSum + noiseSum
Ydiff = diffSum + noiseDiff

lambdaV = 0.5
lambdaGamma = 0.5
lambdaF = 1.0
lambdaPsi=1.0
(V, F, Gamma, Psi, stats_sum, stats_diff) = two_mega_mega_scggm(Ysum*Ydiff, X, Ysum, lambdaV, lambdaF, lambdaGamma, lambdaPsi, r)

plt.figure()
plt.spy(V)
plt.title("V")

plt.figure()
plt.spy(F)
plt.title("F")

plt.figure()
plt.spy(Gamma)
plt.title("Gamma")

plt.figure()
plt.spy(Psi)
plt.title("Psi")

plt.figure()
plt.spy(stats_sum)
plt.title("stats_diff")

plt.figure()
plt.spy(stats_sum)
plt.title("stats_diff")

plt.show()




