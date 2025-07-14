# Executes experiments to get table 1 and 2 in section 6.1 

import experiment_nocovariates

n = experiment_nocovariates.n
grid_size = experiment_nocovariates.grid_size
homoskedastic_sigma = experiment_nocovariates.homoskedastic_sigma 

################### NORMAL THETA CASE ###################
mu_theta = experiment_nocovariates.mu_theta
print("Running normal theta case....") 
problems_normal_theta = experiment_nocovariates.problems_normal_theta
results_normal_case = experiment_nocovariates.make_df_normal(problems_normal_theta, sigma=homoskedastic_sigma, m_sim=50) 
results_normal_case.to_csv('results/nocovariates_normal_homo.csv', index=False) 

################### BINARY THETA CASE ###################
print("Running binary theta case....") 
problems_binary_theta = experiment_nocovariates.problems_binary_theta
results_binary_homoeskedastic_case = experiment_nocovariates.make_df_binary(problems_binary_theta, sigma=homoskedastic_sigma, m_sim=50) 
results_binary_homoeskedastic_case.to_csv('results/no_covariates/binary_homo.csv', index=False) 