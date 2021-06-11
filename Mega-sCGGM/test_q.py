import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ssp
import scipy.sparse.linalg as ssl
import sys

#sys.path.append("../Mega-sCGGM/")
from mega_scggm import mega_scggm
from q_mega_scggm import q_mega_scggm

n = 1000
m = 2
p = 50
q = 50
nan_percent_col=0.2
#r = 5

X=np.random.choice(2,(n,2*p))
print (X.shape)
A=ssp.random(q,q)
Omega=A.T@A+0.01*ssp.identity(q)
Xnew=X.reshape((2*n,-1))
#print(Y.shape)
Xsum= Xnew[::2,:]+Xnew[1::2,:]

#q*q
Delta=ssp.eye(q, format="coo")
Delta.setdiag(0.3, 0)

#p*q
Pi=ssp.rand(p,q)
#p*q
Xi=ssp.rand(p,q)
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
Ysum= Ynew[::2,:]+Ynew[1::2,:]
#print(Y.shape)

for i in range(q):
        l = np.random.choice(np.arange(n), replace=False, size=int(n*nan_percent_col))
        Y[l,i]=float('nan')
        Y[l,i+q]=float('nan')

'''
for i in l:
        Y[:,i]=float('nan')
        Y[:,i+q]=float('nan')

for i in l:                                                                                                         
        indices = np.random.choice(np.arange(q), replace=False, size=int(q * nan_percent_col))
        Y[i,indices]=float("nan")                                                                                               
        Y[i,indices+q]=float("nan")
'''
lambdaV=1.0
lambdaGamma = 1.0
lambdaF = 0.4
lambdaPsi=0.45
(V, F, Gamma, Psi, stats_sum) = q_mega_scggm(Y, X, Ysum, lambdaV, lambdaF, lambdaGamma, lambdaPsi)

re_F=Xi+0.5*Pi
re_Delta=m*Gamma
re_Omega=V-Gamma
re_Xi=F-Psi
re_Pi=m*Psi
TP_x=0
FP_x=0
FN_x=0
TP_f=0
FP_f=0
FN_f=0
TP_o=0
FP_o=0
FN_o=0
TP_p=0
FP_p=0
FN_p=0
Pi_temp=Pi.tocsr()
Xi_temp=Xi.tocsr()
F_temp=re_F.tocsr()
re_F_temp=F.tocsr()
re_Omega_temp=re_Omega.tocsr()
Omega_temp=Omega.tocsr()
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
		if not (math.isclose(F_temp[i,j],0.0) and math.isclose(re_F_temp[i,j],0.0)):

			#if math.isclose(F_temp[i,j],re_F_temp[i,j]):
			if (not math.isclose(F_temp[i,j],0.0)) and (not math.isclose(re_F_temp[i,j],0.0)):
				TP_f+=1;
			elif math.isclose(F_temp[i,j],0.0) and (not math.isclose(re_F_temp[i,j],0.0)):
				FP_f+=1;
			elif (not math.isclose(F_temp[i,j],0.0)) and math.isclose(re_F_temp[i,j],0.0):
				FN_f+=1;
			else:
				print(F_temp[i,j])
				print(re_F_temp[i,j])
				raise Exception("wrong F")


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

for i in range(q):
	for j in range(q):
		if not (math.isclose(Omega_temp[i,j],0.0) and math.isclose(re_Omega_temp[i,j],0.0)):

			#if math.isclose(Omega_temp[i,j],re_Omega_temp[i,j]):
			if (not math.isclose(Omega_temp[i,j],0.0)) and (not math.isclose(re_Omega_temp[i,j],0.0)):
				TP_o+=1;
			elif math.isclose(Omega_temp[i,j],0.0) and (not math.isclose(re_Omega_temp[i,j],0.0)):
				FP_o+=1;
			elif (not math.isclose(Omega_temp[i,j],0.0)) and math.isclose(re_Omega_temp[i,j],0.0):
				FN_o+=1;
			else:
				print(Omega_temp[i,j])
				print(re_Omega_temp[i,j])
				raise Exception("wrong Omega")

	
precision_p=TP_p/(TP_p+FP_p)
recall_p=TP_p/(TP_p+FN_p)

precision_x=TP_x/(TP_x+FP_x)
recall_x=TP_x/(TP_x+FN_x)

precision_f=TP_f/(TP_f+FP_f)
recall_f=TP_f/(TP_f+FN_f)

precision_o=TP_o/(TP_o+FP_o)
recall_o=TP_o/(TP_o+FN_o)

with open('./results-4/precision_recall.txt', 'w') as output_file:
    output_file.write('Pi precision: %f\n' % (precision_p))
    output_file.write('Pi recall: %f\n' % (recall_p))
    output_file.write('Xi precision: %f\n' % (precision_x))
    output_file.write('Xi recall: %f\n' % (recall_x))
    output_file.write('F precision: %f\n' % (precision_f))
    output_file.write('F recall: %f\n' % (recall_f))
    output_file.write('Omega precision: %f\n' % (precision_o))
    output_file.write('Omega recall: %f\n' % (recall_o))
    
    
    output_file.write('%d true Pi appear in estimating Xi\n' % (Pi_in_Xi))
with open('pi.txt','a+') as f:
    f.seek(0)
    c=f.readlines()
    l=np.array(c).astype(np.float)
    x=np.append(l[::2],recall_p).tolist()
    y=np.append(l[1::2],precision_p).tolist()
    plt.figure()
    x_y=list(zip(x,y))
    x_y=sorted(x_y, key = lambda x: x[0])
    x=[i for i,j in x_y]
    y=[j for i,j in x_y]
    plt.plot(x,y,scalex=False, scaley=False)
    plt.xlabel('Pi Recall')
    plt.ylabel('Pi Precision')
    plt.savefig('./results-4/pr_p.png')
    f.seek(2)
    f.write('%f\n' % (recall_p))
    f.write('%f\n' % (precision_p))
with open('xi.txt','a+') as f:
    f.seek(0)    
    l=np.array(f.readlines()).astype(np.float)
    x=np.append(l[::2],recall_x).tolist()
    y=np.append(l[1::2],precision_x).tolist()
    plt.figure()
    x_y=list(zip(x,y))
    x_y=sorted(x_y, key = lambda x: x[0])
    x=[i for i,j in x_y]
    y=[j for i,j in x_y]
    plt.plot(x,y,scalex=False, scaley=False)
    plt.xlabel('Xi Recall')
    plt.ylabel('Xi Precision')
    plt.savefig('./results-4/pr_x.png')
    f.seek(2)
    f.write('%f\n' % (recall_x))
    f.write('%f\n' % (precision_x))

with open('f.txt','a+') as f:
    f.seek(0)
    l=np.array(f.readlines()).astype(np.float)
    x=np.append(l[::2],recall_f).tolist()
    y=np.append(l[1::2],precision_f).tolist()   
    x_y=list(zip(x,y))
    x_y=sorted(x_y, key = lambda x: x[0])
    x=[i for i,j in x_y]
    y=[j for i,j in x_y]
    plt.figure()
    plt.plot(x,y,scalex=False, scaley=False)
    plt.xlabel('F Recall')
    plt.ylabel('F Precision')
    plt.savefig('./results-4/pr_f.png')
    f.seek(2)
    f.write('%f\n' % (recall_f))
    f.write('%f\n' % (precision_f))
with open('o.txt','a+') as f:
    f.seek(0)
    l=np.array(f.readlines()).astype(np.float)
    x=np.append(l[::2],recall_o).tolist()
    y=np.append(l[1::2],precision_o).tolist()
    x_y=list(zip(x,y))
    x_y=sorted(x_y, key = lambda x: x[0])
    x=[i for i,j in x_y]
    y=[j for i,j in x_y]
   
    plt.figure()
    plt.plot(x,y,scalex=False, scaley=False)
    plt.xlabel('Omega Recall')
    plt.ylabel('Oemga Precision')
    plt.savefig('./results-4/pr_o.png')
    f.seek(2)
    f.write('%f\n' % (recall_o))
    f.write('%f\n' % (precision_o))

plt.figure()
#plt.spy(Delta)
temp=plt.imshow(Delta.todense(), cmap='Blues',interpolation="nearest")

plt.colorbar(temp)

plt.title("Delta")
plt.savefig("./results-4/Delta.png")
plt.figure()
#plt.spy(re_Delta)
temp=plt.imshow(np.abs(re_Delta.todense()), cmap='Blues',interpolation="nearest")
# pass this heatmap object into plt.colorbar method.
plt.colorbar(temp)
plt.title("re_Delta")
plt.savefig("./results-4/re_Delta.png")
plt.figure()
Omega.setdiag(0.0,0)
temp=plt.imshow(np.abs(Omega.todense()), cmap='Blues',interpolation="nearest")
# pass this heatmap object into plt.colorbar method.
plt.colorbar(temp)
plt.title("Omega")
plt.savefig("./results-4/Omega.png")
plt.figure()
re_Omega.setdiag(0.0,0)
temp=plt.imshow(np.abs(re_Omega.todense()), cmap='Blues',interpolation="nearest")
# pass this heatmap object into plt.colorbar method.
plt.colorbar(temp)
plt.savefig("./results-4/re_Omega.png")
plt.figure()
temp=plt.imshow(np.abs(Xi.todense()), cmap='Blues',interpolation="nearest")
# pass this heatmap object into plt.colorbar method.
plt.colorbar(temp)
plt.title("Xi")
plt.savefig("./results-4/Xi.png")
plt.figure()
etmp=plt.imshow(np.abs(re_Xi.todense()), cmap='Blues',interpolation="nearest")
# pass this heatmap object into plt.colorbar method.
plt.colorbar(temp)
plt.title("re_Xi")
plt.savefig("./results-4/re_Xi.png")
plt.figure()
temp=plt.imshow(np.abs(Pi.todense()), cmap='Blues',interpolation="nearest")
# pass this heatmap object into plt.colorbar method.
plt.colorbar(temp)
plt.title("Pi")
plt.savefig("./results-4/Pi.png")
plt.figure()
temp=plt.imshow(np.abs(re_Pi.todense()), cmap='Blues',interpolation="nearest")
# pass this heatmap object into plt.colorbar method.
plt.colorbar(temp)
plt.title("re_Pi")

plt.savefig("./results-4/re_Pi.png")

plt.figure()
temp=plt.imshow(np.abs(F.todense()), cmap='Blues',interpolation="nearest")
# pass this heatmap object into plt.colorbar method.
plt.colorbar(temp)
plt.title("F")

plt.savefig("./results-4/F.png")
plt.figure()
temp=plt.imshow(np.abs(re_F.todense()), cmap='Blues',interpolation="nearest")
# pass this heatmap object into plt.colorbar method.
plt.colorbar(temp)
plt.title("re_F")

plt.savefig("./results-4/re_F.png")

