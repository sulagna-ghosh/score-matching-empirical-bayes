import numpy as np 
import pandas as pd 
import torch as tr 
import numpy.random as rn 
from scipy.stats import norm
import math 

# Numerical calculations for the cases when the bayes risk is not closed form 

# Two-point \mu and uniform \sigma 
BR = 0; M = 10; n = 1000000; k = 10
for m in range(M): 
    A_N = rn.uniform(0.1, 0.5, size=(n,)) 
    B_N = rn.binomial(n=1, p=0.5, size=(n,)) 
    theta_N = B_N*(A_N) + (1-B_N)*(k*A_N) 
    sigma_N = np.sqrt(A_N) 
    Z_N = theta_N + sigma_N*rn.normal(size=(n,)) 
    p_N = norm.pdf(Z_N, A_N, np.sqrt(A_N)) / (norm.pdf(Z_N, A_N, np.sqrt(A_N)) + norm.pdf(Z_N, k*A_N, np.sqrt(A_N))) 
    BR_N = (theta_N - A_N*(k-(k-1)*p_N))**2 
    # print(np.mean(BR_N)) 
    BR += np.mean(BR_N) 
print(BR/M) 

# Poisson \mu 
BR = 0; M = 10; B = 100; n = 1000000 
for m in range(M): 
    A_N = rn.uniform(0.1, 1, size=(n,)) 
    theta_N = rn.poisson(lam = 2*A_N) 
    sigma_N = np.sqrt(A_N) 
    Z_N = theta_N + sigma_N*rn.normal(size=(n,)) 
    p_NB = np.zeros((n,B)); e_N = np.zeros(n); p_N = np.zeros(n) 
    for b in range(B): 
        p_NB[:,b] = ((2*A_N)**b) * norm.pdf(Z_N, b*np.ones(n), np.sqrt(A_N)) / math.factorial(b) 
        e_N += b*p_NB[:,b] 
        p_N += p_NB[:,b] 
    BR_N = (theta_N - e_N/p_N)**2 
    # print(np.mean(BR_N)) 
    BR += np.mean(BR_N) 
print(BR/M) 

# Five covariates 
BR = 0; M = 10; n = 1000000 
for m in range(M): 
    A_N = rn.uniform(1.5, 2.5, size=(n,)) 
    X_prev_N5 = rn.uniform(0, 1, size=(n, 5)) 
    m_N = (np.pi*X_prev_N5[:,0]*X_prev_N5[:,1]) + 20*((X_prev_N5[:,2]-0.5)**2) + 5*X_prev_N5[:,3] 
    theta_N = m_N + 2*rn.normal(size=(n,)) 
    sigma_N = np.sqrt(A_N) 
    Z_N = theta_N + sigma_N*rn.normal(size=(n,)) 
    BR_N = (theta_N - (4*Z_N / (4+A_N) + A_N*m_N / (4+A_N)))**2 
    # print(np.mean(BR_N)) 
    BR += np.mean(BR_N) 
print(BR/M) 