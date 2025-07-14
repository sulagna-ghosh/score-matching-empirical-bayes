import train

import torch as tr
import pandas as pd
import numpy as np
import numpy.random as rn

# Contains functions to execute experiments conducted to get table 1 and 2 in section 6.1 

################### GENERAL VARIABLES ###################

n = 1000
grid_size = 5
homoskedastic_sigma = np.ones(n) # Homoscedastic

################### NORMAL THETA CASE ###################

mu_theta = 10

normal_theta = {
    # 'name': "normal_theta",
    'sigma_theta': [np.sqrt(0.1), 1, np.sqrt(5)], 
}

problems_normal_theta = [normal_theta]

################### BINARY THETA CASE ###################

binary_theta_lowk = {
    # 'name': "k = 5",
    'mu': [3, 4, 5, 7],
    'k': 5
}

binary_theta_medk = {
    # 'name': "k = 50",
    'mu': [3, 4, 5, 7],
    'k': 50
}

binary_theta_highk = {
    # 'name': "k = 500",
    'mu': [3, 4, 5, 7],
    'k': 500
}

problems_binary_theta = [binary_theta_lowk, 
                         binary_theta_medk, 
                         binary_theta_highk] 

################### RESULTS ###################

def make_df_normal(problems_normal_theta, sigma, m_sim=50): 

    results = [] 

    for problem in problems_normal_theta: 
    
        for m in range(m_sim):

            print(f"m_sim: {m}") 
            
            problem_copy = problem.copy()

            SURE_2step_MSE = []
            SURE_2step_LOSS = []

            SURE_both_MSE = []
            SURE_both_LOSS = []

            SURE_theta_MSE = []
            SURE_theta_LOSS = []

            SURE_pi_MSE = []
            SURE_pi_LOSS = [] 

            SURE_sparse_MSE = []
            SURE_sparse_LOSS = []

            NPMLE_MSE = []
            NPMLE_LOSS = []
            SURE_NPMLEinit_OBJ_LOSS = []
            SURE_OBJ_LOSS = []

            SURE_NPMLEinit_MSE = []
            SURE_NPMLEinit_LOSS = []

            BAYES_MSE = []

            sigma_thetas = problem_copy['sigma_theta']

            for sigma_theta in sigma_thetas:
                print(f"sigma_theta: {sigma_theta}")
                count = 0
                while count < 1: 
                    try: 
                        ALL_2norm_opt, ALL_loss_opt = train.normal_settings(n, mu_theta, sigma_theta, sigma, B = 100) 
                        
                        SURE_2step_MSE.append(ALL_2norm_opt[4]/n)
                        SURE_2step_LOSS.append(ALL_loss_opt[4]) 

                        SURE_both_MSE.append(ALL_2norm_opt[3]/n) 
                        SURE_both_LOSS.append(ALL_loss_opt[3]) 

                        SURE_theta_MSE.append(ALL_2norm_opt[1]/n) 
                        SURE_theta_LOSS.append(ALL_loss_opt[1]) 

                        SURE_pi_MSE.append(ALL_2norm_opt[2]/n) 
                        SURE_pi_LOSS.append(ALL_loss_opt[2])  

                        SURE_sparse_MSE.append(ALL_2norm_opt[5]/n) 
                        SURE_sparse_LOSS.append(ALL_loss_opt[5]) 

                        NPMLE_MSE.append(ALL_2norm_opt[0]/n) 
                        NPMLE_LOSS.append(ALL_loss_opt[0]) 
                        SURE_NPMLEinit_OBJ_LOSS.append(ALL_loss_opt[7]) 
                        SURE_OBJ_LOSS.append(ALL_loss_opt[8])

                        SURE_NPMLEinit_MSE.append(ALL_2norm_opt[6]/n) 
                        SURE_NPMLEinit_LOSS.append(ALL_loss_opt[6]) 

                        BAYES_MSE.append(ALL_2norm_opt[7]/n)

                        count = count + 1
                    except Exception as e:
                        print(f"Error occurred") 
            
            for new_key in ['SURE_2step_MSE', 'SURE_2step_LOSS', 'SURE_both_MSE', 'SURE_both_LOSS',
                    'SURE_theta_MSE', 'SURE_theta_LOSS', 'SURE_pi_MSE', 'SURE_pi_LOSS', 
                    'SURE_sparse_MSE', 'SURE_sparse_LOSS', 'NPMLE_MSE', 'NPMLE_LOSS', 'SURE_NPMLEinit_OBJ_LOSS', 'SURE_OBJ_LOSS', 
                    'SURE_NPMLEinit_MSE', 'SURE_NPMLEinit_LOSS', 'BAYES_MSE']:
                
                problem_copy[new_key] = eval(new_key) 
            
            results.append(pd.DataFrame(problem_copy)) 
    
    results_df = pd.concat(results)

    return results_df

def make_df_binary(problems_binary_theta, sigma, m_sim=50,
                   use_location=False, use_scale=True): 

    results = [] 

    for problem in problems_binary_theta: 
    
        for m in range(m_sim):

            print(f"m_sim: {m}") 
            
            problem_copy = problem.copy()

            SURE_2step_MSE = []
            SURE_2step_LOSS = []

            SURE_both_MSE = []
            SURE_both_LOSS = []

            SURE_theta_MSE = []
            SURE_theta_LOSS = []

            SURE_pi_MSE = []
            SURE_pi_LOSS = [] 

            SURE_sparse_MSE = []
            SURE_sparse_LOSS = []

            NPMLE_MSE = []
            NPMLE_LOSS = []
            SURE_NPMLEinit_OBJ_LOSS = []
            SURE_OBJ_LOSS = []

            SURE_NPMLEinit_MSE = []
            SURE_NPMLEinit_LOSS = []

            BAYES_MSE = []

            val_thetas = problem_copy['mu']
            k = problem_copy['k']
            
            print(f"k: {k}")

            for val_theta in val_thetas:
            
                print(f"val_theta : {val_theta}")

                count = 0
                while count < 1:
                    try: 
                        ALL_2norm_opt, ALL_loss_opt = train.discrete_settings(n, k, val_theta, sigma, B = 100,
                                                                              use_location=use_location,
                                                                              use_scale=use_scale) 
                        
                        SURE_2step_MSE.append(ALL_2norm_opt[4]/n)
                        SURE_2step_LOSS.append(ALL_loss_opt[4]) 

                        SURE_both_MSE.append(ALL_2norm_opt[3]/n) 
                        SURE_both_LOSS.append(ALL_loss_opt[3]) 

                        SURE_theta_MSE.append(ALL_2norm_opt[1]/n) 
                        SURE_theta_LOSS.append(ALL_loss_opt[1]) 

                        SURE_pi_MSE.append(ALL_2norm_opt[2]/n) 
                        SURE_pi_LOSS.append(ALL_loss_opt[2])  

                        SURE_sparse_MSE.append(ALL_2norm_opt[5]/n) 
                        SURE_sparse_LOSS.append(ALL_loss_opt[5]) 

                        NPMLE_MSE.append(ALL_2norm_opt[0]/n) 
                        NPMLE_LOSS.append(ALL_loss_opt[0]) 
                        SURE_NPMLEinit_OBJ_LOSS.append(ALL_loss_opt[7]) 
                        SURE_OBJ_LOSS.append(ALL_loss_opt[8]) 

                        SURE_NPMLEinit_MSE.append(ALL_2norm_opt[6]/n) 
                        SURE_NPMLEinit_LOSS.append(ALL_loss_opt[6]) 

                        BAYES_MSE.append(ALL_2norm_opt[7]/n) 

                        count = count + 1
                    except Exception as e:
                         print(f"Error occurred") 
            
            for new_key in ['SURE_2step_MSE', 'SURE_2step_LOSS', 'SURE_both_MSE', 'SURE_both_LOSS',
                    'SURE_theta_MSE', 'SURE_theta_LOSS', 'SURE_pi_MSE', 'SURE_pi_LOSS', 
                    'SURE_sparse_MSE', 'SURE_sparse_LOSS', 'NPMLE_MSE', 'NPMLE_LOSS', 'SURE_NPMLEinit_OBJ_LOSS', 'SURE_OBJ_LOSS',
                    'SURE_NPMLEinit_MSE', 'SURE_NPMLEinit_LOSS', 'BAYES_MSE']:
                
                problem_copy[new_key] = eval(new_key) 
            
            results.append(pd.DataFrame(problem_copy)) 
    
    results_df = pd.concat(results)

    return results_df 