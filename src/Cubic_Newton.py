#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from copy import deepcopy
import time
# from FW_shape import FrankWolfe_shape
# from FW_shape import formulate_QU
from utils import *
from Frank_Wolfe import *
# from FW_shape import Line_search




def Cubic_Newton_NDE(Bk, shape, S, lam, c = 1, max_iter = 50, FW_max_iter = 2000, tol = 1e-6, short_steps = 5, logout = True):
    
    '''
    INPUTS:
    
    Bk: a matrix in \R^{n\times M}, evaluation of each base distribution on each data point.
    
    shape: a string, can take the following 9 values:
           "none", "increasing", "decreasing", "convex", "concave", "increasing_convex", "increasing_concave", 
           "decreasing_convex", "decreasing_concave"
           
    S: a vector in \R^{M}, with boolean coordinates.
    
    lam: initial weights on vertices of the constraint set. 
         Let U be the matrix with columns being the vertices of the constraint set and let w be the initial solution, 
         then it holds lam = U @ lam.
    
    c: the regularization parameter for the cubic term.
    
    max_iter: maximal iteration of proximal newton.
    
    FW_max_iter: maximal iteration of FW subproblem.
    
    short_steps: number of short steps taken in the beginning
    
    tol: tolerance of accuracy. Let J be the objective value in current iteration, and J_prev be the objective value in last iteration, then if |J - J_prev| < tol, the algorithm terminates.
    
    RETURNS:
    
    w: the solution
    
    time_vec: a vector in recording the time of each iteration
    
    obj_vec: a vector in recording the objective value of each iteration
    
    FW_iters: a vector in recording the number of FW iterations in each outer iteration
    
    '''
    
    
    t00 = time.time()
    n,M = np.shape(Bk)
    U = formulate_QU(np.eye(M), shape)
    w = U @ lam
    f = -(1/n)*np.sum(np.log(Bk.dot(w)))
    ns = 5*M
    
    idx = np.round(np.cumsum(np.ones(n)))
    obj_vec = np.zeros(max_iter)
    time_vec = np.zeros(max_iter)
    FW_iters = np.zeros(max_iter)
    
    threshold = -1
    n_short = 0
    
    
    for k in range(max_iter):
        
        t1 = time.time()
        
        g_hat = Bk.dot(w)
        R =  Bk/(g_hat.reshape(-1,1)+1e-10*np.ones(g_hat.reshape(-1,1).shape,dtype=float))
        R = (R > 1e-10) * R
        grad = -(1/n)*np.sum(R,axis=0).reshape(1,-1).reshape(-1)
        Hessian = (1/n)*(R.transpose()).dot(R)
        
        t2 = time.time()
        
        if k<= 3:
            tol_FW = 1e-8
            
        if k<=6 and k>=4:
            tol_FW = 1e-9
            
        if k>=10:
            tol_FW = 1e-10
        
        w_prev = w
        lam_prev = lam
        iterations, w, S, lam = FrankWolfe_shape(Q = 0.5*deepcopy(Hessian),
                                          a = deepcopy(grad),
                                          x0 = deepcopy(w),
                                          c = c,
                                          max_iter = 2000,
                                          stepsize = True,
                                          S = deepcopy(S),
                                          tol = tol_FW, 
                                          x = deepcopy(w),
                                          U = deepcopy(U),
                                          lam0 = lam,
                                          shape=shape)
        w = w * (w>0)
        t3 = time.time()
        f_prev = f
        g = Bk.dot(w)
        f = -(1/n)*np.sum(np.log(Bk.dot(w)))

        second_order_term = 0.5* (Hessian.dot(w - w_prev)).dot(w - w_prev)
        check = f_prev + grad.dot(w - w_prev) + second_order_term + c* (np.sqrt(second_order_term))**3 - f
        
        threshold = threshold*0.8
        if check < threshold:
            c = c * 1.5

        w_half = w_prev + 0.5*(w - w_prev)
        lam_half = lam_prev + 0.5*(lam-lam_prev)
        f_half = -(1/n)*np.sum(np.log(Bk.dot(w_half)))

        if k<=short_steps:
            if f - f_half> threshold:
                w = w_half
                lam = lam_half
                S = np.ones((M))
                if shape == "convex":
                    S = np.ones((2*M))
                f = f_half
                n_short = n_short + 1   

        t4 = time.time()
        obj_vec[k] = f
        time_vec[k] = t4 - t00
        FW_iters[k] = iterations
        
        
        
        if logout == True:
            print("----------------------------------")
            print( "k=",k, " obj=", obj_vec[k], "FW_iters=", iterations )
        
        if np.abs(f - f_prev) < tol:
                return w, time_vec[0:k+1], obj_vec[0:k+1], FW_iters[0:k+1], lam, S

        
        
    return w, time_vec, obj_vec, FW_iters, lam, S














