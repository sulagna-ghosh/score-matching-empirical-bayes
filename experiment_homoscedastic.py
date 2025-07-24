import train

import torch as tr
import pandas as pd
import numpy as np
import numpy.random as rn

from train import train_npmle, train_no_covariates
import simulate_data
import models

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


def normal_settings(n, mu_theta, sigma_theta, sigma, B, 
                      use_location=False, use_scale=True):
    '''
    Returns MSE and SURE loss for no covariates normal settings, possibly using location and scale if needed. 
    '''

    # Simulate data
    theta, Z, X = simulate_data.simulate_data_normal_nocovariates(n, mu_theta, sigma_theta, sigma)

    # NPMLE 
    results_NPMLE = train_npmle(n, B, Z, theta, X) 
    problem_NPMLE, loss_NPMLE, score_NPMLE, theta_hat_NPMLE, twonorm_diff_NPMLE, pi_hat_NPMLE = results_NPMLE
    pi_hat_NPMLE[pi_hat_NPMLE < 0] = 0

    # EB (only theta)
    results_EB_theta = train_no_covariates(n, B, Z, theta, X, opt_objective = 'theta-only',
                                           use_location=use_location, use_scale=use_scale) 
    model_EB_theta, SURE_EB_theta, scores_EB_theta, theta_hats_EB_theta, twonorm_diff_EB_theta = results_EB_theta
    SURE_EB_theta = SURE_EB_theta[-1]
    theta_diff_EB_theta = model_EB_theta.forward() 

    # EB (only pi)
    results_EB_pi = train_no_covariates(n, B, Z, theta, X, opt_objective = 'pi-only',
                                           use_location=use_location, use_scale=use_scale)
    model_EB_pi, SURE_EB_pi, scores_EB_pi, theta_hats_EB_pi, twonorm_diff_EB_pi = results_EB_pi
    SURE_EB_pi = SURE_EB_pi[-1]

    # EB (both theta and pi)
    results_EB_both = train_no_covariates(n, B, Z, theta, X, opt_objective = 'both',
                                           use_location=use_location, use_scale=use_scale)
    model_EB_both, SURE_EB_both, scores_EB_both, theta_hats_EB_both, twonorm_diff_EB_both = results_EB_both
    SURE_EB_both = SURE_EB_both[-1]

    # EB (two-step)
    results_EB_2step = train_no_covariates(n, B, Z, theta, X, opt_objective = 'both', 
                                                 init_val_theta = tr.log(theta_diff_EB_theta).to('cpu'),
                                           use_location=use_location, use_scale=use_scale)
    model_EB_2step, SURE_EB_2step, scores_EB_2step, theta_hats_EB_2step, twonorm_diff_EB_2step = results_EB_2step
    SURE_EB_2step = SURE_EB_2step[-1]

    # EB (sparse)
    results_EB_sparse = train_no_covariates(n, B, Z, theta, X, opt_objective = 'pi-sparse',
                                           use_location=use_location, use_scale=use_scale) 
    model_EB_sparse, SURE_EB_sparse, scores_EB_sparse, theta_hats_EB_sparse, twonorm_diff_EB_sparse = results_EB_sparse
    SURE_EB_sparse = SURE_EB_sparse[-1]

    # EB (both pi and theta; NPMLE initial)
    results_EB_both_npmle = train_no_covariates(n, B, Z, theta, X, opt_objective = 'both', init_val_pi = tr.log(pi_hat_NPMLE),
                                           use_location=use_location, use_scale=use_scale)
    model_EB_both_npmle, SURE_EB_both_npmle, scores_EB_both_npmle, theta_hats_EB_both_npmle, twonorm_diff_EB_both_npmle = results_EB_both_npmle 
    SURE_EB_both_npmle = SURE_EB_both_npmle[-1]

    # EB (only pi; NPMLE initial; one iteration)
    model = models.model_pi_sure(Z, B, init_val=tr.log(pi_hat_NPMLE), device=simulate_data.device) 
    SURE_EB_pi_npmle_one_iter = model.opt_func(Z, n, B, sigma = X[:,-1]) 
    SURE_NPMLE_OBJ = SURE_EB_pi_npmle_one_iter.item() 

    # EB (loss for one iteration with uniform initialization for pi)
    model = models.model_pi_sure(Z, B, init_val=tr.log(tr.Tensor([1.5])), device=simulate_data.device) 
    SURE_EB_pi_one_iter = model.opt_func(Z, n, B, sigma = X[:,-1]) 
    SURE_EB_OBJ = SURE_EB_pi_one_iter.item() 

    # BAYES 
    sigma = X[:,-1] 
    theta_hat_bayes = mu_theta*(sigma**2)/(sigma**2+sigma_theta**2) + Z*(sigma_theta**2)/(sigma**2+sigma_theta**2)
    twonorm_diff_BAYES = (np.linalg.norm(theta_hat_bayes.cpu().detach().numpy() - theta)**2) 

    # List of 2-norm differences and optimal losses
    ALL_2norm_opt = [twonorm_diff_NPMLE, twonorm_diff_EB_theta, twonorm_diff_EB_pi, twonorm_diff_EB_both, twonorm_diff_EB_2step, 
                     twonorm_diff_EB_sparse, twonorm_diff_EB_both_npmle, twonorm_diff_BAYES]
    ALL_loss_opt = [loss_NPMLE, SURE_EB_theta, SURE_EB_pi, SURE_EB_both, SURE_EB_2step, SURE_EB_sparse, SURE_EB_both_npmle, 
                    SURE_NPMLE_OBJ, SURE_EB_OBJ]

    return (ALL_2norm_opt, ALL_loss_opt) 

def discrete_settings(n, k, val_theta, sigma, B, 
                      use_location=False, use_scale=True):
    '''
    Returns MSE and SURE loss for no covariates discrete settings, possibly using location and scale if needed. 
    '''

    # Simulate data
    theta, Z, X = simulate_data.simulate_data_discrete_nocovariates(n, k, val_theta, sigma)

    # NPMLE 
    results_NPMLE = train_npmle(n, B, Z, theta, X) 
    problem_NPMLE, loss_NPMLE, score_NPMLE, theta_hat_NPMLE, twonorm_diff_NPMLE, pi_hat_NPMLE = results_NPMLE
    pi_hat_NPMLE[pi_hat_NPMLE < 0] = 0

    # EB (only theta)
    results_EB_theta = train_no_covariates(n, B, Z, theta, X, opt_objective = 'theta-only',
                                           use_location=use_location, use_scale=use_scale) 
    model_EB_theta, SURE_EB_theta, scores_EB_theta, theta_hats_EB_theta, twonorm_diff_EB_theta = results_EB_theta
    SURE_EB_theta = SURE_EB_theta[-1]
    theta_diff_EB_theta = model_EB_theta.forward() 

    # EB (only pi)
    results_EB_pi = train_no_covariates(n, B, Z, theta, X, opt_objective = 'pi-only',
                                           use_location=use_location, use_scale=use_scale)
    model_EB_pi, SURE_EB_pi, scores_EB_pi, theta_hats_EB_pi, twonorm_diff_EB_pi = results_EB_pi
    SURE_EB_pi = SURE_EB_pi[-1]

    # EB (both theta and pi)
    results_EB_both = train_no_covariates(n, B, Z, theta, X, opt_objective = 'both',
                                           use_location=use_location, use_scale=use_scale)
    model_EB_both, SURE_EB_both, scores_EB_both, theta_hats_EB_both, twonorm_diff_EB_both = results_EB_both
    SURE_EB_both = SURE_EB_both[-1]

    # EB (two-step)
    results_EB_2step = train_no_covariates(n, B, Z, theta, X, opt_objective = 'both', 
                                                 init_val_theta = tr.log(theta_diff_EB_theta).to('cpu'),
                                           use_location=use_location, use_scale=use_scale)
    model_EB_2step, SURE_EB_2step, scores_EB_2step, theta_hats_EB_2step, twonorm_diff_EB_2step = results_EB_2step
    SURE_EB_2step = SURE_EB_2step[-1]

    # EB (sparse)
    results_EB_sparse = train_no_covariates(n, B, Z, theta, X, opt_objective = 'pi-sparse',
                                           use_location=use_location, use_scale=use_scale) 
    model_EB_sparse, SURE_EB_sparse, scores_EB_sparse, theta_hats_EB_sparse, twonorm_diff_EB_sparse = results_EB_sparse
    SURE_EB_sparse = SURE_EB_sparse[-1]

    # EB (both pi and theta; NPMLE initial)
    results_EB_both_npmle = train_no_covariates(n, B, Z, theta, X, opt_objective = 'both', init_val_pi = tr.log(pi_hat_NPMLE).to('cpu'),
                                           use_location=use_location, use_scale=use_scale)
    model_EB_both_npmle, SURE_EB_both_npmle, scores_EB_both_npmle, theta_hats_EB_both_npmle, twonorm_diff_EB_both_npmle = results_EB_both_npmle 
    SURE_EB_both_npmle = SURE_EB_both_npmle[-1] 

    # EB (only pi; NPMLE initial; one iteration)
    model = models.model_pi_sure(Z, B, init_val=tr.log(pi_hat_NPMLE), device=simulate_data.device)
    SURE_EB_pi_npmle_one_iter = model.opt_func(Z, n, B, sigma = X[:,-1]) 
    SURE_NPMLE_OBJ = SURE_EB_pi_npmle_one_iter.item() 

    # EB (loss for one iteration with uniform initialization for pi)
    model = models.model_pi_sure(Z, B, init_val=tr.log(tr.Tensor([1.5])), device=simulate_data.device)
    SURE_EB_pi_one_iter = model.opt_func(Z, n, B, sigma = X[:,-1]) 
    SURE_EB_OBJ = SURE_EB_pi_one_iter.item() 

    # BAYES 
    # TODO: Fix the Bayes solution. No more Bernoulli
    sigma = X[:,-1]
    theta_num = val_theta*(k/n)*tr.exp(-0.5*(((Z-val_theta)/sigma)**2)) 
    theta_denom = (k/n)*tr.exp(-0.5*(((Z-val_theta)/sigma)**2)) + (1-k/n)*tr.exp(-0.5*((Z/sigma)**2)) 
    theta_hat_bayes = theta_num / theta_denom 
    twonorm_diff_BAYES = (np.linalg.norm(theta_hat_bayes.cpu().detach().numpy() - theta)**2) 

    # List of 2-norm differences and optimal losses
    ALL_2norm_opt = [twonorm_diff_NPMLE, twonorm_diff_EB_theta, twonorm_diff_EB_pi, twonorm_diff_EB_both, twonorm_diff_EB_2step, 
                     twonorm_diff_EB_sparse, twonorm_diff_EB_both_npmle, twonorm_diff_BAYES]
    ALL_loss_opt = [loss_NPMLE, SURE_EB_theta, SURE_EB_pi, SURE_EB_both, SURE_EB_2step, SURE_EB_sparse, SURE_EB_both_npmle, 
                    SURE_NPMLE_OBJ, SURE_EB_OBJ]

    return (ALL_2norm_opt, ALL_loss_opt) 

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
                        ALL_2norm_opt, ALL_loss_opt = normal_settings(n, mu_theta, sigma_theta, sigma, B = 100) 
                        
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
                        ALL_2norm_opt, ALL_loss_opt = discrete_settings(n, k, val_theta, sigma, B = 100,
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