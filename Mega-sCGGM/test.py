import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ssp
import scipy.sparse.linalg as ssl
import sys

#sys.path.append("../Mega-sCGGM/")
from mega_scggm import mega_scggm
from two_mega_scggm import two_mega_scggm

n = 100
m = 2
p = 50
q = 50
#r = 5

X=np.random.choice(2,(n,2*p))
print (X.shape)
A=ssp.random(q,q,density=0.03)
Omega=A.T@A+0.01*ssp.identity(q)

#q*q
Delta=ssp.eye(q, format="coo")
Delta.setdiag(0.3, 1)
Delta.setdiag(0.3, -1)
#p*q
Pi=ssp.rand(p,q,density=0.02)
#r*q
Xi=ssp.rand(p,q,density=0.02)
Xnew=X.reshape((2*n,-1))
#print(Y.shape)
Xsum= Xnew[::2,:]+Xnew[1::2,:]
#
Lambda=ssp.kron(ssp.identity(m,format="coo"),Delta)+ssp.kron(np.ones((m,m)),Omega)
Theta=ssp.vstack([ssp.kron(ssp.identity(m,format="coo"),Pi),ssp.kron(np.ones((1,m)),Xi)])

Sigma = ssl.inv(Lambda)
meanY = -1 * np.hstack((X,Xsum)) @ Theta @ Sigma
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
##print(Y)
#print(Ysum)
lambdaV=0.45
lambdaGamma = 1.1
lambdaF = 0.75
lambdaPsi=0.6
(V, F, Gamma, Psi, stats_sum, stats_diff) = two_mega_scggm(Y, X, Xsum, lambdaV, lambdaF, lambdaGamma, lambdaPsi)

re_Delta=m*Gamma
re_Omega=V-Gamma
re_Xi=F-Psi
re_Pi=m*Psi
#plt.figure()
#plt.spy(V)
#plt.title("V")
#plt.savefig("./results/V.png")

plt.figure()
plt.spy(Delta)
plt.title("Delta")
plt.savefig("./results/Delta.png")
plt.figure()
plt.spy(re_Delta)
plt.title("re_Delta")
plt.savefig("./results/re_Delta.png")
plt.figure()
plt.spy(Omega)
plt.title("Omega")
plt.savefig("./results/Omega.png")
plt.figure()
plt.spy(re_Omega)
plt.title("re_Omega")
plt.savefig("./results/re_Omega.png")
plt.figure()
plt.spy(Xi)
plt.title("Xi")
plt.savefig("./results/Xi.png")
plt.figure()
plt.spy(re_Xi)
plt.title("re_Xi")
plt.savefig("./results/re_Xi.png")
plt.figure()
plt.spy(Pi)
plt.title("Pi")
plt.savefig("./results/Pi.png")
plt.figure()
plt.spy(re_Pi)
plt.title("re_Pi")

plt.savefig("./results/re_Pi.png")
