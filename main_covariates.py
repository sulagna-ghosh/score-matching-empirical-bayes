
import experiment_covariates_highD

import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import torch as tr
import numpy.random as rn

d = experiment_covariates_highD.d; m = experiment_covariates_highD.m; mu_x = experiment_covariates_highD.mu_x
print(f"d: {d}")
print(f"m: {m}")
print(f"mu_x: {mu_x}") 

################### MISSPECIFIED: drop sigma ###################
print("################### MISSPECIFIED: drop sigma ###################")
# extra information case (heteroskedastic)
problems_dropsigma = experiment_covariates_highD.problems_dropsigma
# print(f"problems_dropsigma: {problems_dropsigma}")
results_df_dropsigma = experiment_covariates_highD.make_df(problems_dropsigma, m_sim=3, drop_sigma=True)
results_df_dropsigma.to_csv('results/covariates_dropsigma.csv', index=False) 


################## WELL SPECIFIED CASE ###################
print("################## WELL SPECIFIED CASE ###################")
problems = experiment_covariates_highD.problems
# print(f"problems: {problems}")
results_df = experiment_covariates_highD.make_df(problems, m_sim=3)
results_df.to_csv('results/covariates_wellspecified.csv', index=False) 

################## CHECKING FT_REP ###################


# n = 5
# B = 100
# d = 3 # m(X) is at most, quadratic
# m = 4
# mu_x = np.array([0.5, 3, 2, 3])

# B_dmplus1 = np.zeros([d,m+1]); B_dmplus1[0,0] = 10; B_dmplus1[1,:-1] = 1; B_dmplus1[1,-1] = 0.25
# B_dmplus1[2,] = -0.2 # low correlation

# sigma=rn.uniform(1.5, 3, size=(n,))
# print(f"sigma: {sigma}")

# X, m_X, sigma_x, theta, sigma_theta, Z, m, d, n = simulate_data.simulate_data(n=n, mu_x=mu_x, sigma_x = np.ones((m,)), 
#                                                                                     sigma=sigma, m=m, d=d, B_dmplus1=B_dmplus1)


# model = train_NN.modelNN(X, d=d, B=B, hidden_sizes=(256, 256), init_std=0.01, init_val=tr.Tensor([1.5]))
# Ft_Rep = model.feature_representation(X)
# print(f"Ft_Rep: {Ft_Rep}")
# print(f"Ft_Rep.shape: {Ft_Rep.shape}")
# print(f"Check last 2 column: {Ft_Rep[:,-3:-1]}")
# print(f"X last column: {X[:,-1]}")


# model_mis = train_misspecified_NN.modelNN(X, d=d, B=B, hidden_sizes=(256, 256), init_std=0.01, init_val=tr.Tensor([1.5]))
# Ft_Rep_mis = model_mis.feature_representation(X)
# print(f"Ft_Rep_mis: {Ft_Rep_mis}")
# print(f"Ft_Rep_mis.shape: {Ft_Rep_mis.shape}")
# print(f"Last column is indeed sigma: {Ft_Rep_mis[:,-2:-1]}")