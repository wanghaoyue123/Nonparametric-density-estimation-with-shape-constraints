#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from copy import deepcopy
import time









def grad_h(t, xQx, dQx, dQd, ad, c):
    # h(t):= (d'Qd)t^2 + (2(x'Qd)+a'd)t + c( (x'Qx) + 2(d'Qx)t + (d'Qd)t^2 )^{3/2}
    # h'(t) = 2(d'Qd)t + (2(x'Qd)+a'd) + 3*c( (x'Qx) + 2(d'Qx)t + (d'Qd)t^2 )^{1/2} * ((d'Qx)+ (d'Qd)t)
    
    val = 2* dQd * t + (2* dQx + ad) + 3* c* np.sqrt(xQx + 2*dQx*t + dQd*t**2) * (dQx + dQd*t)
    
    return val


def Line_search(Q = None,
                a = None,
                y = None,
                d = None,
                c = None,
                Qy = None,
                Qd = None,
                tol = 1e-12
               ):
    # y = x - x0
    # min_{t\in [0,1]}  (y+td)'Q(y+td) + < a, y+td > + c((y+td)'Q(y+td))^{3/2}  
    # 
    # or equivalently,
    # 
    # min_{t\in [0,1]}   h(t):= (d'Qd)t^2 + (2(y'Qd)+a'd)t + c( (y'Qy) + 2(d'Qy)t + (d'Qd)t^2 )^{3/2}
    # 
    #       h'(t) = 2(d'Qd)t + (2(y'Qd)+a'd) + 3*c( (y'Qy) + 2(d'Qy)t + (d'Qd)t^2 )^{1/2} * ((d'Qy)+ (d'Qd)t)
    
    ub = 1
    lb = 0
    yQy = y.dot(Qy)
    dQy = d.dot(Qy)
    dQd = d.dot(Qd)
    ad = a.dot(d)
    
    if grad_h(0, yQy, dQy, dQd, ad, c) > 0:
        
        return 0
    
    if grad_h(1, yQy, dQy, dQd, ad, c) < 0:

        return 1
        
    itermax = 100
    
    t = 0.5*(ub + lb)
    for i in range(itermax):
        t = 0.5*(ub + lb)
        gd_h = grad_h(t, yQy, dQy, dQd, ad, c)
        
        if np.abs(gd_h)< tol:
            return t
        else:
            if gd_h > 0:
                ub = t
            else:
                lb = t
    
    return t







def formulate_QU(Q, shape):
    
    mm = np.shape(Q)
    m = mm[0]
    M = m
    inv_idx = [m-i-1 for i in range(m)]
    inv_idx2 = [m-i-2 for i in range(m-1)]
    
    if shape == "none":
        return Q
    
    
    if shape == "decreasing":
        QU = np.zeros((m, m))
        QU = np.cumsum(Q, axis=1)
        D = 1/np.cumsum(np.ones((m,)))
        QU = QU*D
        return QU
    
    if shape == "increasing":
        QU = np.zeros((m, m))
        QU = np.cumsum(Q[:, ::-1], axis=1)[:, ::-1]
        D = 1/np.cumsum(np.ones((m,)))[::-1]
        QU = QU*D
        return QU
        
    if shape == "increasing_concave":
        a = 0
        a = Q[0,0]
        q1 = np.zeros((m,))
        q1 = Q[0, 1:m]
        q2 = np.zeros((m,))
        q2 = Q[1:m, 0]
        
        _Q = np.zeros((m,m))
        _Q = Q[1:m, 1:m]
        
        QT = np.zeros((m,m))
        QT[0,0] = a + np.sum(q1)
        QT[0, 1:m] = np.cumsum(np.cumsum(q1[::-1])[::-1])
        QT[1:m, 0] = q2 + np.sum(_Q, axis = 1)
        QT[1:m, 1:m] = np.cumsum(np.cumsum(_Q[:, ::-1], axis=1)[:, ::-1], axis = 1)
        
        D = np.zeros((m,))
        D[0] = 1/m
        D[1:m] = 1/(m* np.cumsum(np.ones(m-1)) - np.cumsum(np.cumsum(np.ones(m-1))))
        QU = QT * D
        return QU
        
    if shape == "decreasing_concave":
        
        QQ = deepcopy(Q[:, ::-1])
        return formulate_QU(QQ, "increasing_concave")
        
        
    if shape == "increasing_convex":
        q1 = Q[:,0]
        Q2 = Q[:, 1:m]
        QT = np.zeros((m,m))
        QT[:, 0:m-1] = np.cumsum(np.cumsum(Q2[:, ::-1], axis = 1), axis = 1)
        QT[:, m-1] = q1 + np.sum(Q2, axis = 1)
        
        D = np.zeros((m,))
        D[0:m-1] = 1/np.cumsum(np.cumsum(np.ones(m-1)))
        D[m-1] = 1/m
        QU = QT * D
        return QU
        
    if shape == "decreasing_convex":
        
        QQ = deepcopy(Q[:, ::-1])
        return formulate_QU(QQ, "increasing_convex")
    
    
    if shape == "concave":
        QT_1 = np.zeros((m,m))
        QT_1 = np.cumsum(np.cumsum(Q, axis = 1)[:, ::-1], axis =1)[:, ::-1]
        
        QT_2 = np.zeros((m,m))
        tmp1 = np.zeros((m,2))
        tmp1[:, 0] = QT_1[:, 0]
        tmp1[:, 1] = QT_1[:, m-1]
        MM = (1/(m-1))* np.array([[1,-1], [-1,m]])
        tmp2 = tmp1 @ MM
        
        BT = np.zeros((2,m))
        BT[0,1] = 1
        BT[1,m-2] = 1
        BT[1,m-1] = -1
        tmp3 = np.zeros((2,m))
        tmp3 = np.cumsum(np.cumsum(BT, axis = 1)[:, ::-1], axis =1)[:, ::-1]
        QT_2 = tmp2@tmp3
        
        QT = QT_1 - QT_2
        
        D = np.zeros((m,))
        D[0] = 2/m
        D[m-1] = 2/m
        D[1:m-1] = 1/(  0.5*m*np.cumsum(np.ones(m-2))  - np.cumsum(np.cumsum(np.ones(m-2)))  )
        
        
        QU = QT*D
        return QU
    
    
    if shape == "convex":
        D = 1/np.cumsum(np.cumsum(np.ones(m)))
        QT1 = np.cumsum(np.cumsum(Q[:, ::-1], axis=1), axis=1)
        QT2 = np.cumsum(np.cumsum(Q, axis=1),  axis=1)

        
        QU = np.zeros((m,2*m))
        QU[:, 0:m] = QT1*D
        QU[:, m:2*m] = QT2*D

        return QU
    
    
    return 
        
        

        
        
        
        
        
        
        

 
def UT_dot(x, shape):
    
    mm = np.shape(x)
    m = mm[0]
    M = m
    inv_idx = [m-i-1 for i in range(m)]
    
    if shape == "none":
        return x
    
    if shape == "decreasing":
        UTx = np.zeros((m, ))
        D = 1/np.cumsum(np.ones((m,)))
        UTx = np.cumsum(x)
        UTx = D * UTx
        return UTx
    
    if shape == "increasing":
        UTx = np.zeros((m, ))
        D = 1/np.cumsum(np.ones((m,)))[::-1]
        TTx = np.cumsum(x[::-1])[::-1]
        UTx = D * TTx
        return UTx
        
    if shape == "increasing_concave":
        a= x[0]
        _x = x[1:m]
        
        TTx = np.zeros((m, ))
        TTx[0] = a + np.sum(_x)
        TTx[1:m] = np.cumsum( np.cumsum(_x[::-1])[::-1] )
        
        D = np.zeros((m,))
        D[0] = 1/m
        D[1:m] = 1/(m* np.cumsum(np.ones(m-1)) - np.cumsum(np.cumsum(np.ones(m-1))))
        UTx = D * TTx
        return UTx
    
    if shape == "decreasing_concave":
        
        xx = deepcopy(x[::-1])
        return UT_dot(xx, "increasing_concave")
    
    
    if shape == "increasing_convex":

        x1 = x[0]
        _x = x[1:m]
        

        TTx = np.zeros((m, ))
        TTx[0:m-1] = np.cumsum(np.cumsum(_x[::-1]))
        TTx[m-1] = x1 + np.sum(_x)
        

        D = np.zeros((m,))
        D[0:m-1] = 1/np.cumsum(np.cumsum(np.ones(m-1)))
        D[m-1] = 1/m
        UTx = TTx * D
        

        return UTx
    
    if shape == "decreasing_convex":
        xx = deepcopy(x[::-1])
        return UT_dot(xx, "increasing_convex")
    
    
    if shape == "concave":
        T1_x = np.cumsum(np.cumsum(x)[::-1])[::-1]
        
        tmp1 = np.array([T1_x[0], T1_x[m-1]])
        MM = (1/(M-1))* np.array([[1,-1], [-1,m]])
        tmp2 = MM @ tmp1
        tmp3 = np.zeros((m,))
        tmp3[1] = tmp2[0]
        tmp3[m-2] = tmp2[1]
        tmp3[m-1] = -tmp2[1]
        T2_x = np.cumsum(np.cumsum(tmp3)[::-1])[::-1]
        
        TTx = T1_x - T2_x
        
        D = np.zeros((m,))
        D[0] = 2/m
        D[m-1] = 2/m
        D[1:m-1] = 1/(  0.5*m*np.cumsum(np.ones(m-2))  - np.cumsum(np.cumsum(np.ones(m-2)))  )
        
        UTx = TTx * D
        
        return UTx
    
    
    if shape == "convex":
        TTx1 = np.cumsum(np.cumsum(x[::-1]))
        TTx2 = np.cumsum(np.cumsum(x))
        D = 1/np.cumsum(np.cumsum(np.ones(m)))
        UTx = np.zeros(2*m)
        UTx[0:m] = TTx1 * D
        UTx[m:2*m] = TTx2 * D
        
        return UTx
    
    return
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
    
    print("shape=", np.shape(S))

    
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


