#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from copy import deepcopy
import time
from utils import *


    
    
    
    
def FrankWolfe_shape(Q=None, # Hessian
                           a=None, # gradient
                           x0=None, 
                           c=None , # regularization parameter
                           max_iter = 10000,
                           stepsize=True,
                           x = None,
                           S = None, # Support set 
                           tol=1e-8, 
                           U = None,
                           lam0 = None,
                           shape = None
                           ): 
    ############################################################
    # min  (x-x0)'Q(x-x0) + <a,x-x0> + c ((x-x0)'Q(x-x0))^(3/2)
    # s.t. x\in U(\Delta_K)
    #
    # The columns of U are the vertices of the constraint set.
    #
    #
    #         
    #
    ###################################################################
    
    K = Q.shape[0]
    if shape == "convex":
        K = 2*K
    
    QU = formulate_QU(Q, shape)
    obj = np.zeros(max_iter,dtype=float)
    inv_idx = range(K - 1, -1, -1)
    diag_D = np.array([1/(K-i) for i in range(K)])
    diag_D2= np.array([K-i for i in range(K)])
    ## lam is a weight vector in \R^K that records the weight of each vertex on x
    lam = lam0
    y = x - x0
    Qx = Q.dot(x)
    Qx0 = Q.dot(x0)
    Qy = Qx - Qx0
    yQy = y.dot(Qy)
    

    
    for it in range(max_iter):
        
        grad = 2*(Qy) + a + 3*c* np.sqrt(yQy)*Qy
        
        ## compute U.T @ grad
        t1 = time.time()
        UT_grad = UT_dot(grad, shape)
        t2 = time.time()
        
        # FW direction
        s = np.zeros((K,),dtype=float)
        idx_FW = np.argmin(UT_grad)
        s = U[:, idx_FW]
        delta_FW = s - x        
        
        # Away direction
        v = np.zeros((K,),dtype=float)
        S_idx = (S>0.5).nonzero()[0]
        idx_A = S_idx[np.argmax(UT_grad[S_idx])]
        v = U[:, idx_A]
        delta_A = x - v 
        
        t3 = time.time()


        # Update x and S 
        if  -grad.dot(delta_FW) >= -grad.dot(delta_A): # Choose FW direction
            delta = deepcopy(delta_FW)
            Qd = QU[:, idx_FW] - Qx
            gamma = Line_search(Q, a, y, delta, c, Qy, Qd, tol = 1e-12)
            x += gamma*delta
            Qx = (1-gamma)*Qx + gamma* QU[:, idx_FW]
            if gamma==1.0:
                S = np.zeros(K)
                S[idx_FW] = 1
            else:
                S[idx_FW] = 1
            lam = lam* (1-gamma) 
            lam[idx_FW] = lam[idx_FW] + gamma
                    
        else: # Choose away direction; maximum feasible step-size
            delta = deepcopy(delta_A)
            gamma_max = lam[idx_A]/(1-lam[idx_A])
            Qd = (Qx - QU[:, idx_A])*gamma_max
            sst = time.time()
            t = Line_search(Q, a, y, gamma_max*delta, c, Qy, Qd, tol = 1e-12)
            eed = time.time()
            gamma = gamma_max * t
            x += gamma* delta
            Qx = (1+gamma)*Qx - gamma* QU[:, idx_A]
            
            if abs(gamma - gamma_max)<1e-50:
                S[idx_A] = 0

            lam = lam* (1+ gamma) 
            lam[idx_A] = lam[idx_A] - gamma
        
        t4 = time.time()
        
        y = x - x0
        Qy = Qx - Qx0
        yQy = y.dot(Qy)
        obj[it] = yQy + a.dot(y) + c*(yQy)**(1.5)
        
        t5 = time.time()
        
        
        if np.absolute(obj[it] - obj[it-1])/(np.maximum(np.absolute(obj[it-1]),1)) <= tol and it>10:
            break    
            
    return it+1, x, S, lam


