# Executes experiments to get table 1 and 2 in section 6.1 

import experiment_nocovariates

n = experiment_nocovariates.n
grid_size = experiment_nocovariates.grid_size
homoskedastic_sigma = experiment_nocovariates.homoskedastic_sigma
heteroskedastic_sigma = experiment_nocovariates.heteroskedastic_sigma

################### NORMAL THETA CASE ###################
# mu_theta = experiment_nocovariates.mu_theta
# print("################### NORMAL THETA CASE ###################")
# problems_normal_theta = experiment_nocovariates.problems_normal_theta
# results_normal_case = experiment_nocovariates.make_df_normal(problems_normal_theta, m_sim=50) 
# results_normal_case.to_csv('results/nocovariates_normal_homo.csv', index=False) 

################### BINARY THETA CASE ###################
# print("################### BINARY THETA CASE ###################")
problems_binary_theta = experiment_nocovariates.problems_binary_theta
results_binary_homoeskedastic_case = experiment_nocovariates.make_df_binary(problems_binary_theta, sigma=homoskedastic_sigma, m_sim=1) 
results_binary_homoeskedastic_case.to_csv('results/no_covariates/binary_homo.csv', index=False) 

results_binary_heteroskedastic_case = experiment_nocovariates.make_df_binary(problems_binary_theta, sigma=heteroskedastic_sigma, m_sim=1) 
results_binary_heteroskedastic_case.to_csv('results/no_covariates/binary_hetero.csv', index=False) 

################### NORMAL THETA CASE (Short version) ###################
# mu_theta = experiment_nocovariates.mu_theta
# print("################### NORMAL THETA CASE ###################")
# problems_normal_theta = experiment_nocovariates.problems_normal_theta
# results_normal_case = experiment_nocovariates.make_df_normal_main(ns = [10000], m_sim = 25) 
# results_normal_case.to_csv('results/nocovariates_normal_short.csv', index=False) 

################### BINARY THETA CASE (Short version) ###################
# print("################### BINARY THETA CASE ###################")
# problems_binary_theta = experiment_nocovariates.problems_binary_theta
# results_binary_case = experiment_nocovariates.make_df_binary_main(ns = [1000, 2500, 5000], m_sim = 5) 
# results_binary_case.to_csv('results/nocovariates_binary_short.csv', index=False) 

################### NORMAL THETA CASE (B COMPARISON) ###################
# mu_theta = experiment_nocovariates.mu_theta
# print("################### NORMAL THETA CASE ###################")
# problems_normal_theta = experiment_nocovariates.problems_normal_theta
# results_normal_case_B = experiment_nocovariates.make_df_normal_B(problems_normal_theta, m_sim=50) 
# results_normal_case_B.to_csv('results/nocovariates_normal_B.csv', index=False) 

################### BINARY THETA CASE (B COMPARISON) ###################
# print("################### BINARY THETA CASE ###################")
# problems_binary_theta = experiment_nocovariates.problems_binary_theta
# results_binary_case_B = experiment_nocovariates.make_df_binary_B(problems_binary_theta, m_sim=50) 
# results_binary_case_B.to_csv('results/nocovariates_binary_B.csv', index=False) 