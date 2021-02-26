import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ssp
import scipy.sparse.linalg as ssl
import sys

#sys.path.append("../Mega-sCGGM/")
from mega_scggm import mega_scggm
from two_mega_scggm import two_mega_scggm

n = 4
m = n//2
p = 5
q = 5
r = 1

X=np.random.randn(n,m*p+r)
Omega=ssp.csr_matrix(np.array([[1,1,0,0,0],[1,1,1,0,0],[0,1,1,0,0],[0,0,0,0,0],[0,0,0,0,0]]))

#q*q
Delta=ssp.eye(q, format="coo")
Delta.setdiag(0.3, 1)
Delta.setdiag(0.3, -1)
#p*q
Pi=ssp.rand(p,q)
#r*q
Xi=ssp.rand(r,q)

Lambda=ssp.kron(ssp.identity(m,format="coo"),Delta)+ssp.kron(np.ones((m,m)),Omega)
Theta=ssp.vstack([ssp.kron(ssp.identity(m,format="coo"),Pi),ssp.kron(np.ones((1,m)),Xi)])

Sigma = ssl.inv(Lambda)
meanY = -1 * X @ Theta @ Sigma
'''
try:
    import sksparse.cholmod as skc
    Lambda_factor = skc.cholesky(Lambda)
    noiseY = (Lambda_factor.solve_Lt(np.random.randn(m*q,n))).transpose()
except:
'''
noiseY = np.random.multivariate_normal(np.zeros(m*q), Sigma.todense(), size=n)
#n * mq 4*10
Y = meanY + noiseY
# 8*5
(n_y, q_prime) = Y.shape
Ynew=Y.reshape((2*n_y,-1))
#print(Y.shape)
Ysum= Ynew[::2,:]+Ynew[1::2,:]
#print(Y)
#print(Ysum)
lambdaV=0.5
lambdaGamma = 0.5
lambdaF = 1.0
lambdaPsi=1.0
(V, F, Gamma, Psi, stats_sum, stats_diff) = two_mega_scggm(Y, X, Ysum, lambdaV, lambdaF, lambdaGamma, lambdaPsi, r)

re_Delta=m*Gamma
re_Omega=V-Gamma
re_Xi=(ssl.inv(Pi.T)@m*F.T).T
re_Pi=m*Psi

plt.figure()
plt.spy(Delta)
plt.title("Delta")

plt.figure()
plt.spy(re_Delta)
plt.title("re_Delta")

plt.figure()
plt.spy(Omega)
plt.title("Omega")

plt.figure()
plt.spy(re_Omega)
plt.title("re_Omega")

plt.figure()
plt.spy(Xi)
plt.title("Xi")

plt.figure()
plt.spy(re_Xi)
plt.title("re_Xi")

plt.figure()
plt.spy(Pi)
plt.title("Pi")

plt.figure()
plt.spy(re_Pi)
plt.title("re_Pi")

plt.show()