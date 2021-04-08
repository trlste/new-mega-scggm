import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ssp
import scipy.sparse.linalg as ssl
import sys

#sys.path.append("../Mega-sCGGM/")
from mega_scggm import mega_scggm
from q_mega_scggm import q_mega_scggm

n = 500
m = 2
p = 50
q = 50
nan_percent_col=0.05
#r = 5

X=np.random.choice(2,(n,2*p))
print (X.shape)
A=ssp.random(q,q,density=0.02)
Omega=A.T@A+0.01*ssp.identity(q)
Xnew=X.reshape((2*n,-1))
#print(Y.shape)
Xsum= Xnew[::2,:]+Xnew[1::2,:]

#q*q
Delta=ssp.eye(q, format="coo")
Delta.setdiag(0.3, 0)

#p*q
Pi=ssp.rand(p,q,density=0.02)
#p*q
Xi=ssp.rand(p,q,density=0.02)
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
Ynew=Y.reshape((2*n,-1))
#print(Y.shape)
Ysum= Ynew[::2,:]+Ynew[1::2,:]
for i in range(n):                                                                                                          
	indices = np.random.choice(np.arange(q), replace=False, size=int(q * nan_percent_col))                                  
	Y[i,indices]=float("nan")                                                                                               
	Y[i,indices+q]=float("nan")
# 8*5
##print(Y)
#print(Ysum)
lambdaV=0.45
lambdaGamma = 1.1
lambdaF = 0.75
lambdaPsi=0.6
(V, F, Gamma, Psi, stats_sum) = q_mega_scggm(Y, X, Ysum, lambdaV, lambdaF, lambdaGamma, lambdaPsi)

re_Delta=m*Gamma
re_Omega=V-Gamma
re_Xi=F-Psi
re_Pi=m*Psi
#plt.figure()
#plt.spy(V)
#plt.title("V")
#plt.savefig("./results/V.png")
TP_x=0
FP_x=0
FN_x=0

TP_p=0
FP_p=0
FN_p=0
Pi_temp=Pi.tocsr()
Xi_temp=Xi.tocsr()
re_Pi_temp=re_Pi.tocsr()
re_Xi_temp=re_Xi.tocsr()
Pi_in_Xi=0
for i in range(p):
	for j in range(q):
		if not (math.isclose(Pi_temp[i,j],0.0) and math.isclose(re_Pi_temp[i,j],0.0)):

			#if math.isclose(Pi_temp[i,j],re_Pi_temp[i,j]):
			if (not math.isclose(Pi_temp[i,j],0.0)) and (not math.isclose(re_Pi_temp[i,j],0.0)):
				TP_p+=1;
			elif math.isclose(Pi_temp[i,j],0.0) and (not math.isclose(re_Pi_temp[i,j],0.0)):
				FP_p+=1;
			elif (not math.isclose(Pi_temp[i,j],0.0)) and math.isclose(re_Pi_temp[i,j],0.0):
				FN_p+=1;
			else:
				print(Pi_temp[i,j])
				print(re_Pi_temp[i,j])
				raise Exception("wrong Pi")

		if not (math.isclose(Xi_temp[i,j],0.0) and math.isclose(re_Xi_temp[i,j],0.0)):
			if (not math.isclose(Xi_temp[i,j],0.0)) and (not math.isclose(re_Xi_temp[i,j],0.0)):
			#if math.isclose(Xi_temp[i,j],re_Xi_temp[i,j]):
				TP_x+=1;
			elif math.isclose(Xi_temp[i,j],0.0) and (not math.isclose(re_Xi_temp[i,j],0.0)):
				FP_x+=1;
			elif (not math.isclose(Xi_temp[i,j],0.0)) and math.isclose(re_Xi_temp[i,j],0.0):
				FN_x+=1;
			else:
				raise Exception("wrong Xi")
	
		if math.isclose(re_Xi_temp[i,j],Pi_temp[i,j]):
			if not math.isclose(Pi_temp[i,j],0.0):
				Pi_in_Xi+=1;


precision_p=TP_p/(TP_p+FP_p)
recall_p=TP_p/(TP_p+FN_p)

precision_x=TP_x/(TP_x+FP_x)
recall_x=TP_x/(TP_x+FN_x)

with open('./results/precision_recall.txt', 'w') as output_file:
    output_file.write('Pi precision: %f\n' % (precision_p))
    output_file.write('Pi recall: %f\n' % (recall_p))
    output_file.write('Xi precision: %f\n' % (precision_x))
    output_file.write('Xi recall: %f\n' % (recall_x))
    output_file.write('%d true Pi appear in estimating Xi\n' % (Pi_in_Xi))

plt.figure()
#plt.spy(Delta)
temp=plt.imshow(Delta.todense(), cmap='Blues',interpolation="nearest")

plt.colorbar(temp)

plt.title("Delta")
plt.savefig("./results/Delta.png")
plt.figure()
#plt.spy(re_Delta)
temp=plt.imshow(np.abs(re_Delta.todense()), cmap='Blues',interpolation="nearest")
# pass this heatmap object into plt.colorbar method.
plt.colorbar(temp)
plt.title("re_Delta")
plt.savefig("./results/re_Delta.png")
plt.figure()
Omega.setdiag(0.0,0)
temp=plt.imshow(np.abs(Omega.todense()), cmap='Blues',interpolation="nearest")
# pass this heatmap object into plt.colorbar method.
plt.colorbar(temp)
plt.title("Omega")
plt.savefig("./results/Omega.png")
plt.figure()
re_Omega.setdiag(0.0,0)
temp=plt.imshow(np.abs(re_Omega.todense()), cmap='Blues',interpolation="nearest")
# pass this heatmap object into plt.colorbar method.
plt.colorbar(temp)
plt.savefig("./results/re_Omega.png")
plt.figure()
temp=plt.imshow(np.abs(Xi.todense()), cmap='Blues',interpolation="nearest")
# pass this heatmap object into plt.colorbar method.
plt.colorbar(temp)
plt.title("Xi")
plt.savefig("./results/Xi.png")
plt.figure()
etmp=plt.imshow(np.abs(re_Xi.todense()), cmap='Blues',interpolation="nearest")
# pass this heatmap object into plt.colorbar method.
plt.colorbar(temp)
plt.title("re_Xi")
plt.savefig("./results/re_Xi.png")
plt.figure()
temp=plt.imshow(np.abs(Pi.todense()), cmap='Blues',interpolation="nearest")
# pass this heatmap object into plt.colorbar method.
plt.colorbar(temp)
plt.title("Pi")
plt.savefig("./results/Pi.png")
plt.figure()
temp=plt.imshow(np.abs(re_Pi.todense()), cmap='Blues',interpolation="nearest")
# pass this heatmap object into plt.colorbar method.
plt.colorbar(temp)
plt.title("re_Pi")

plt.savefig("./results/re_Pi.png")
plt.figure()
temp=plt.imshow(np.abs(F.todense()), cmap='Blues',interpolation="nearest")
# pass this heatmap object into plt.colorbar method.
plt.colorbar(temp)
plt.title("F")

plt.savefig("./results/F.png")
