#!/bin/python

import os
import numpy as np
import numpy.random as rnd
import scipy.sparse as ssp
import sys

sys.path.append("../Python/")
from txt_to_sparse import txt_to_sparse
from txt_to_dict import txt_to_dict
from sparse_to_txt import sparse_to_txt

def two_mega_scggm(
        Y, X, Y_sum, lambdaV, lambdaF, lambdaGamma, lambdaPsi, r,
        verbose=False, max_iters=50, sigma=1e-4, tol=1e-2,
        num_blocks_V=-1, num_blocks_F=-1, num_blocks_Gamma=-1,
        num_blocks_Psi=-1, memory_usage=32000,
        threads=16, refit=False, V0=None, F0=None, Gamma0=None,Psi0=None):
    """
    Args:
      (mscggm)Y: output data matrix (n samples x q dimensions target variables)
      (mscggm)X: input data matrix (n samples x p dimensions covariate variables)
      X:n*(2p+r)
      Y:(n*2q) with missing inputs
      Ysum: n*q
      lambdaV: regularization for V
      lambdaF: regularization for F
      lambdaGamma: regularization for Gamma
      lambdaPsi: regularization for Psi
      r: the number of group-level inputs

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
      F0: p+r x q scipy.sparse matrix to initialize F
      Gamma0: q x q scipy.sparse matrix to initialize Gamma
      Psi0: p x q scipy.sparse matrix to initialize Psi

    Returns:
        V: q x q sparse matrix
        F: p+r x q sparse matrix
        Gamma: q x q sparse matrix
        Psi: p x q sparse matrix
        stats_sum: dict of logging results (of the summation term)
        stats_diff: dict of logging results (of the difference term)
    """

    olddir = os.getcwd()
    thisdir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(thisdir) # move to demo directory
    os.chdir("../Mega-sCGGM")

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

    
    #n_y=n, q_prime=2q
    #n_x=n, p_prime=2p+r
    (n_y, q_prime) = Y.shape 
    (n_x, p_prime) = X.shape

    #2n*q
    Y=Y.reshape((2*n_y,-1))

    q=q_prime//2
    p=(p_prime-r)//2
    #n*2p->2n*p
    X_r_dropped=X[:,:2*p].reshape((2*n_x,-1))
    X_r_dropped_p=X_r_dropped[::2,:]
    X_r_dropped_m=X_r_dropped[1::2,:]
    #n*r    
    X_r=X[:,2*p:]
    #n*p+r
    X_sum=np.concatenate((X_r_dropped_m+X_r_dropped_p,X_r),axis=1)
    #n*p
    X_diff=np.subtract(X_r_dropped_p, X_r_dropped_m)

    # calculates ys and xs
    
    #means 2n*q
    Y_means=np.repeat(.5*Y_sum, repeats=2, axis=0)
    Y_complete= np.where(np.isnan(Y),Y_means,Y)
   
    Y_p=Y_complete[::2,:]
    Y_m=Y_complete[1::2,:]
    Y_diff=np.subtract(Y_p,Y_m)

    Y_sum_file = "Y_sum-dummy-%i.txt" % (dummy)
    X_sum_file = "X_sum-dummy-%i.txt" % (dummy)
    
    Y_diff_file= "Y_diff-dummy-%i.txt" % (dummy)
    X_diff_file= "X_diff-dummy-%i.txt" % (dummy)

    Vfile = "V-dummy-%i.txt" % (dummy)
    Ffile = "F-dummy-%i.txt" % (dummy)
    Gammafile = "Gamma-dummy-%i.txt" % (dummy)
    Psifile = "Psi-dummy-%i.txt" % (dummy)

    stats_sum_file = "stats_sum-dummy-%i.txt" % (dummy)
    stats_diff_file = "stats_diff-dummy-%i.txt" % (dummy)
  
    np.savetxt(Y_sum_file, Y_sum, fmt="%.10f", delimiter=" ")
    np.savetxt(X_sum_file, X_sum, fmt="%.10f", delimiter=" ")
    np.savetxt(Y_diff_file, Y_diff, fmt="%.10f", delimiter=" ")
    np.savetxt(X_diff_file, X_diff, fmt="%.10f", delimiter=" ")

    #First call to mega-scggm: V and F, the summation term
    mega_str = "-l %i -t %i -m %i -n %i " % (
        num_blocks_V, num_blocks_F, memory_usage, threads)
    option_str = "-y %f -x %f -v %i -i %i -s %f -q %f -r %i  %s  %s %s " % (
        lambdaV, lambdaF,
        verbose, max_iters, sigma, tol, refit, 
        mega_str, V0_str, F0_str)
    command_str_sum = "./mega_scggm %s   %i %i %i %i %s %s   %s %s %s" % (
        option_str,
        n_y, q, n_x, p+r, Y_sum_file, X_sum_file,
        Vfile, Ffile, stats_sum_file)
    print(command_str_sum)

    #second call to mega-scggm: Gamma and Psi, the difference term
    mega_str = "-l %i -t %i -m %i -n %i " % (
        num_blocks_Gamma, num_blocks_Psi, memory_usage, threads)
    option_str = "-y %f -x %f -v %i -i %i -s %f -q %f -r %i  %s  %s %s " % (
        lambdaGamma, lambdaPsi,
        verbose, max_iters, sigma, tol, refit, 
        mega_str, G0_str, P0_str)
    command_str_diff = "./mega_scggm %s   %i %i %i %i %s %s   %s %s %s" % (
        option_str,
        n_y, q, n_x, p, Y_diff_file, X_diff_file,
        Gammafile, Psifile, stats_diff_file)
    print(command_str_diff)

    #simply calls mega-scggm twice
    ret_sum = os.system(command_str_sum)
    ret_diff = os.system(command_str_diff)

    V = txt_to_sparse(Vfile)
    F = txt_to_sparse(Ffile)
    Gamma = txt_to_sparse(Gammafile)
    Psi = txt_to_sparse(Psifile)
    stats_sum = txt_to_dict(stats_sum_file)
    stats_diff = txt_to_dict(stats_diff_file)
    rmline = "rm %s %s %s %s %s %s %s %s" % (Yfile, Xfile, Vfile, Ffile, Gammafile, Psifile, stats_sum_file, stats_diff_file)

    ret = os.system(rmline)
    os.chdir(olddir)
    return (V, F, Gamma, Psi, stats_sum, stats_diff)

