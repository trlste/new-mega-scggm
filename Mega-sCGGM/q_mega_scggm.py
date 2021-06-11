#!/bin/pythoni

import os
import numpy as np
import numpy.random as rnd
import scipy.sparse as ssp
import sys
import time
sys.path.append("../Python/")
from txt_to_sparse import txt_to_sparse
from txt_to_dict import txt_to_dict
from sparse_to_txt import sparse_to_txt

def q_mega_scggm(
        Y, X, Y_sum, lambdaV, lambdaF, lambdaGamma, lambdaPsi,
        verbose=False, max_iters=50, sigma=1e-4, tol=1e-2,
        num_blocks_V=-1, num_blocks_F=-1, num_blocks_Gamma=-1,
        num_blocks_Psi=-1, memory_usage=32000,
        threads=16, refit=False, V0=None, F0=None, Gamma0=None,Psi0=None):
    """
    Args:
      (mscggm)Y: output data matrix (n samples x q dimensions target variables)
      (mscggm)X: input data matrix (n samples x p dimensions covariate variables)
      X:n*(2p)
      Y:n*2q
      
      lambdaV: regularization for V
      lambdaF: regularization for F
      lambdaGamma: regularization for Gamma
      lambdaPsi: regularization for Psi

    Optional args:
      verbose: print information or notnew
      max_iters: max number of outer iterations
      sigma: backtracking termination criterion
      tol: tolerance for terminating outer loop
      num_blocks_V: number of blocks for V CD
      num_blocks_F: number of blocks for F CD
       num_blocks_Gamma: number of blocks for Gamma CD
      num_blocks_Psi: number of blocks for Psi CD
      memory_usage: memory capacity in MB
      threads: the maximum number of threads
      refit: refit (Lambda0, Theta0) without adding any edges
      V0: q x q scipy.sparse matrix to initialize V
      F0: p x q scipy.sparse matrix to initialize F
      Gamma0: q x q scipy.sparse matrix to initialize Gamma
      Psi0: p x q scipy.sparse matrix to initialize Psi

    Returns:
        V: q x q sparse matrix
        F: p x q sparse matrix
        Gamma: q x q sparse matrix
        Psi: p x q sparse matrix
        stats_sum: dict of logging results (of the summation term)
        stats_diff: dict of logging results (of the difference term)
    """

    olddir = os.getcwd()
    thisdir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(thisdir) # move to demo directory
    os.chdir("../Mega-sCGGM")
    t0=time.time()
    dummy = rnd.randint(low=0, high=1e6)
    V0_str = ""
    F0_str = ""
    G0_str = ""
    P0_str= ""
    if V0:
        V0file = "V0-dummy-%i.txt" % (dummy)
        sparse_to_txt(V0file, V0)
        V0_str = "-L \"%s\" " % (V0file)
    if F0:
        F0file = "F0-dummy-%i.txt" % (dummy)
        sparse_to_txt(F0file, F0)
        T0_str = "-T \"%s\" " % (F0file)
    if Gamma0:
        G0file = "G0-dummy-%i.txt" % (dummy)
        sparse_to_txt(G0file, Gamma0)
        G0_str = "-L \"%s\" " % (G0file)
    if Psi0:
        P0file = "P0-dummy-%i.txt" % (dummy)
        sparse_to_txt(P0file, Psi0)
        P0_str = "-T \"%s\" " % (P0file)

    (n_y, q_p) = Y.shape
    (n_x, p_p) = X.shape
    
    Y=Y.reshape((2*n_y,-1))
    X=X.reshape((2*n_x,-1))
    q=q_p//2
    p=p_p//2

    X_p=X[::2,:]
    X_m=X[1::2,:]
    X_sum=X_p+X_m

    Y_sum_file = "Y_sum-dummy-%i.txt" % (dummy)
    X_sum_file = "X_sum-dummy-%i.txt" % (dummy)
    Vfile = "V-dummy-%i.txt" % (dummy)
    Ffile = "F-dummy-%i.txt" % (dummy)
    stats_sum_file = "stats_sum-dummy-%i.txt" % (dummy)
    np.savetxt(Y_sum_file, Y_sum, fmt="%.10f", delimiter=" ")
    np.savetxt(X_sum_file, X_sum, fmt="%.10f", delimiter=" ")
    



    Y_p=Y[::2,:]
    Y_m=Y[1::2,:]
    X_diff=np.subtract(X_p,X_m)
    Y_diff=np.subtract(Y_p,Y_m)
    
    t1=time.time()
    print("preprocessing time is: "+str((t1-t0)//60))
    #First call to mega-scggm: V and F, the summation term
    
    mega_str = "-l %i -t %i -m %i -n %i " % (
        num_blocks_V, num_blocks_F, memory_usage, threads)
    option_str = "-y %f -x %f -v %i -i %i -s %f -q %f -r %i  %s  %s %s " % (
        lambdaV, lambdaF,
        verbose, max_iters, sigma, tol, refit, 
        mega_str, V0_str, F0_str)
    command_str_sum = "./mega_scggm %s   %i %i %i %i %s %s   %s %s %s" % (
        option_str,
        n_y, q, n_x, p, Y_sum_file, X_sum_file,
        Vfile, Ffile, stats_sum_file)
    print(command_str_sum)
    ret_sum = os.system(command_str_sum) 
    
    t2=time.time()
    print("sum time is: "+str((t2-t1)//60))
    
    V = txt_to_sparse(Vfile)
    F = txt_to_sparse(Ffile)
    stats_sum = txt_to_dict(stats_sum_file)
    
    Gamma=ssp.identity(q,format="lil")
    Psi=ssp.lil_matrix((p,q))
    
    t3=time.time()
    print("postprocessing time is: "+str((t3-t2)//60))
    
    for i in range(q):
        dummy = rnd.randint(low=0, high=1e6)
        Y_diff_file= "Y_diff-dummy-%i.txt" % (dummy)
        X_diff_file= "X_diff-dummy-%i.txt" % (dummy)
        
        curr_Y=Y_diff[:,i][np.logical_not(np.isnan(Y_diff[:,i]))].reshape(-1,1)
        (n_curr_y,_)=curr_Y.shape
        print(n_curr_y)
        if(n_curr_y==0):
            continue

        np.savetxt(Y_diff_file, curr_Y, fmt="%.10f", delimiter=" ")
        np.savetxt(X_diff_file, X_diff, fmt="%.10f", delimiter=" ")
        Gammafile = "Gamma-dummy-%i.txt" % (dummy)
        Psifile = "Psi-dummy-%i.txt" % (dummy)
        stats_diff_file = "stats_diff-dummy-%i.txt" % (dummy)

        #second call to mega-scggm: Gamma and Psi, the difference term
        mega_str = "-l %i -t %i -m %i -n %i " % (
            num_blocks_Gamma, num_blocks_Psi, memory_usage, threads)
        option_str = "-y %f -x %f -v %i -i %i -s %f -q %f -r %i  %s  %s %s " % (
            lambdaGamma, lambdaPsi,
            verbose, max_iters, sigma, tol, refit, 
            mega_str, G0_str, P0_str)
        command_str_diff = "./mega_scggm %s   %i %i %i %i %s %s   %s %s %s" % (
            option_str,
            n_curr_y, 1, n_x, p, Y_diff_file, X_diff_file,
            Gammafile, Psifile, stats_diff_file)
    #    print(command_str_diff)
        ret_diff = os.system(command_str_diff)
        #1*1
        try:
            gamma_i = txt_to_sparse(Gammafile)
            Gamma[i,i]=gamma_i.tolil()[0,0]
        except:
            Gamma[i,i]=0.0
        #p*1
        try:
            psi_i = txt_to_sparse(Psifile)
            Psi[:,i]=psi_i.tolil()
        except:
            pass
        rmline= "rm %s %s %s %s %s" % (Gammafile, Psifile, stats_diff_file, Y_diff_file, X_diff_file)
        ret=os.system(rmline)




    
    t4=time.time()
    print("diff time is: "+str((t4-t3)//60))
    
    Gamma=Gamma.tocoo()
    Psi=Psi.tocoo()

    rmline = "rm %s %s %s %s %s" % (Y_sum_file, X_sum_file, Vfile, Ffile,stats_sum_file)
    ret = os.system(rmline)
    os.chdir(olddir)
    return (V,F,Gamma, Psi,stats_sum)
