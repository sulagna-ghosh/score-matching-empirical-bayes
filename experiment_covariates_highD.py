import simulate_data
import train

import numpy as np
import numpy.random as rn
import pandas as pd
import matplotlib.pyplot as plt
# import copy # deep copy

# Defines functions for executing different cases for SURE-THING under homo/heteroscedastic case with more than one covariates 

################### WELL SPECIFIED CASE ###################

# define problems

d = 3 # m(X) is at most, quadratic
m = 10
mu_x = np.array([0.5, 3, 2, 3, 4, 1, 3, 1, 3, 4])

B_dmplus1_highcorr = np.zeros([d,m+1]); B_dmplus1_highcorr[0,0] = 10; B_dmplus1_highcorr[1,:-1] = 1; B_dmplus1_highcorr[1,-1] = 0.25
B_dmplus1_highcorr[2,] = 0.2

homoskedastic_zhighcorrtheta = {
    'name': "homoskedastic_zhighcorrtheta",
    'sigma': [1, 3], 
    'B_dmplus1': B_dmplus1_highcorr
}

heteroskedastic_zhighcorrtheta = {
    'name': "heteroskedastic_zhighcorrtheta",
    'sigma': ["rn.uniform(0.5, 1.5, size=(n,))", "rn.uniform(1.5, 3, size=(n,))"], 
    'B_dmplus1': B_dmplus1_highcorr
}

B_dmplus1_lowcorr = np.zeros([d,m+1]); B_dmplus1_lowcorr[0,0] = 10; B_dmplus1_lowcorr[1,:-1] = 1; B_dmplus1_lowcorr[1,-1] = 0.25
B_dmplus1_lowcorr[2,:] = -0.2

homoskedastic_zlowcorrtheta = {
    'name': "homoskedastic_zlowcorrtheta",
    'sigma': [1, 3], 
    'B_dmplus1': B_dmplus1_lowcorr
}

heteroskedastic_zlowcorrtheta = {
    'name': "heteroskedastic_zlowcorrtheta",
    'sigma': ["rn.uniform(0.5, 1.5, size=(n,))", "rn.uniform(1.5, 3, size=(n,))"], 
    'B_dmplus1': B_dmplus1_lowcorr
}

problems = [homoskedastic_zhighcorrtheta,
            homoskedastic_zlowcorrtheta,
            heteroskedastic_zhighcorrtheta,
            heteroskedastic_zlowcorrtheta]

################### MISSPECIFIED: drop sigma ###################

# make the coefficients on sigma larger
B_dmplus1_highcorr_dropsigma = np.copy(B_dmplus1_highcorr)
B_dmplus1_lowcorr_dropsigma = np.copy(B_dmplus1_lowcorr)
B_dmplus1_highcorr_dropsigma[:,-1] = -1
B_dmplus1_lowcorr_dropsigma[:,-1] = -1

heteroskedastic_zhighcorrtheta_dropsigma = {
    'name': "heteroskedastic_zhighcorrtheta_dropsigma",
    'sigma': ["rn.uniform(0.5, 1.5, size=(n,))", "rn.uniform(1.5, 3, size=(n,))"], 
    'B_dmplus1': B_dmplus1_highcorr_dropsigma
}

heteroskedastic_zlowcorrtheta_dropsigma = {
    'name': "heteroskedastic_zlowcorrtheta_dropsigma",
    'sigma': ["rn.uniform(0.5, 1.5, size=(n,))", "rn.uniform(1.5, 3, size=(n,))"], 
    'B_dmplus1': B_dmplus1_lowcorr_dropsigma
}

problems_dropsigma = [heteroskedastic_zhighcorrtheta_dropsigma, heteroskedastic_zlowcorrtheta_dropsigma]


def get_simulation_result(problem, m=10, d=3, mu_x=mu_x, B=100, n=1000, save_plots=False, drop_sigma=False):
    """
    inputs:
    * problem is a dictionary with keys:
    """

    B_dmplus1 = problem['B_dmplus1']
    # print(f"B_dmplus1: {B_dmplus1}")

    # initialize sigmas
    if isinstance(problem['sigma'][0], str):
        sigmas = []
        for i in range(len(problem['sigma'])):
            sigmas.append(eval(problem['sigma'][i]))
    elif isinstance(problem['sigma'][0], int):
        sigmas = problem['sigma']
    else:
        print('error!!!')

    rhos = []
    BayesRisk = []
    empirical_BayesRisk = []
    MLE_MSE = []

    EB_SURE = []
    EB_NLL = []
    EB_MSE = []

    NPMLE_SURE = []
    NPMLE_NLL = []
    NPMLE_MSE = []

    if save_plots:
        fig,ax = plt.subplots()  #create a new figure

    if drop_sigma:
            
        for sigma, idx in zip(sigmas, range(len(problem['sigma']))):
            # print(f"sigma: {sigma}")
            # print(f"idx: {idx}")

            X, m_X, sigma_x, theta, sigma_theta, Z, m, d, n = simulate_data.simulate_data_covariates(n=n, mu_x=mu_x, sigma_x = np.ones((m,)), 
                                                                                        sigma=sigma, m=m, d=d, B_dmplus1=B_dmplus1)
            
            rho_Ztheta = np.corrcoef(Z.detach().numpy(), theta)[0,1]
            # print(f"rho_Ztheta: {rho_Ztheta}")

            if save_plots:
                ax.scatter(Z.detach().numpy(), theta, label=problem['sigma'][idx], alpha=0.5)
                plt.xlabel(r"MLE")
                plt.ylabel(r"$\theta_i$")
                ax.legend(loc = 'best')
                fig.savefig("covariates/data_" + problem['name'] + '.png')

            bayes_risk, t_Bayes, empirical_bayesrisk = simulate_data.bayes_things(X, m_X, theta, sigma_theta, Z, n)
            # print(f"t_Bayes : {t_Bayes}")
            # print(f"bayes_risk : {bayes_risk}")
            # print(f"empirical_bayesrisk : {empirical_bayesrisk}")
            # print(f"MLE MSE: {np.linalg.norm(theta - Z.detach().numpy())**2 / n}")

            model_EB, Ft_Rep_EB, losses_SURE_EB, losses_NLL_EB, scores_EB, theta_hats_EB, two_norm_differences_EB = train.train_covariates(X, Z, theta, objective="SURE", B=B, 
                                                                                                                                          drop_sigma=drop_sigma, d=d)
            losses_SURE_EB = losses_SURE_EB[-1]
            losses_NLL_EB = losses_NLL_EB[-1]
            # print(f"EB SURE: {losses_SURE_EB[-1]/n}")
            # print(f"EB NLL: {losses_NLL_EB[-1]/n}")
            # print(f"EB MSE: {np.linalg.norm(theta_hats_EB[-1,:] - theta)**2 / n}")

            # NPMLE
            model_NPMLE, Ft_Rep_NPMLE, losses_SURE_NPMLE, losses_NLL_NPMLE, scores_NPMLE, theta_hats_NPMLE, two_norm_differences_NPMLE = train.train_covariates(X, Z, theta, objective="NLL", B=B, 
                                                                                                                                                                drop_sigma=drop_sigma)
            losses_SURE_NPMLE = losses_SURE_NPMLE[-1]
            losses_NLL_NPMLE = losses_NLL_NPMLE[-1]
            # print(f"NPMLE SURE: {losses_SURE_NPMLE[-1]/n}")
            # print(f"NPMLE NLL: {losses_NLL_NPMLE[-1]/n}")
            # print(f"NPMLE MSE: {np.linalg.norm(theta_hats_NPMLE[-1,:] - theta)**2 / n}

            rhos.append(rho_Ztheta)
            BayesRisk.append(bayes_risk)
            empirical_BayesRisk.append(empirical_bayesrisk)
            MLE_MSE.append(np.linalg.norm(theta - Z.detach().numpy())**2 / n)

            EB_SURE.append(losses_SURE_EB)
            EB_NLL.append(losses_NLL_EB)
            EB_MSE.append(two_norm_differences_EB / n)

            NPMLE_SURE.append(losses_SURE_NPMLE)
            NPMLE_NLL.append(losses_NLL_NPMLE)
            NPMLE_MSE.append(two_norm_differences_NPMLE / n)

    else:

        for sigma, idx in zip(sigmas, range(len(problem['sigma']))):
            # print(f"sigma: {sigma}")
            # print(f"idx: {idx}")

            X, m_X, sigma_x, theta, sigma_theta, Z, m, d, n = simulate_data.simulate_data_covariates(n=n, mu_x=mu_x, sigma_x = np.ones((m,)), 
                                                                                        sigma=sigma, m=m, d=d, B_dmplus1=B_dmplus1)
            
            rho_Ztheta = np.corrcoef(Z.detach().numpy(), theta)[0,1]
            # print(f"rho_Ztheta: {rho_Ztheta}")

            if save_plots:
                ax.scatter(Z.detach().numpy(), theta, label=problem['sigma'][idx], alpha=0.5)
                plt.xlabel(r"MLE")
                plt.ylabel(r"$\theta_i$")
                ax.legend(loc = 'best')
                fig.savefig("covariates/data_" + problem['name'] + '.png')

            bayes_risk, t_Bayes, empirical_bayesrisk = simulate_data.bayes_things(X, m_X, theta, sigma_theta, Z, n)
            # print(f"t_Bayes : {t_Bayes}")
            # print(f"bayes_risk : {bayes_risk}")
            # print(f"empirical_bayesrisk : {empirical_bayesrisk}")
            # print(f"MLE MSE: {np.linalg.norm(theta - Z.detach().numpy())**2 / n}")

            model_EB, Ft_Rep_EB, losses_SURE_EB, losses_NLL_EB, scores_EB, theta_hats_EB, two_norm_differences_EB = train.train_covariates(X, Z, theta, objective="SURE", B=B)
            losses_SURE_EB = losses_SURE_EB[-1]
            losses_NLL_EB = losses_NLL_EB[-1]
            # print(f"EB SURE: {losses_SURE_EB[-1]/n}")
            # print(f"EB NLL: {losses_NLL_EB[-1]/n}")
            # print(f"EB MSE: {np.linalg.norm(theta_hats_EB[-1,:] - theta)**2 / n}")

            # NPMLE
            model_NPMLE, Ft_Rep_NPMLE, losses_SURE_NPMLE, losses_NLL_NPMLE, scores_NPMLE, theta_hats_NPMLE, two_norm_differences_NPMLE = train.train_covariates(X, Z, theta, objective="NLL", B=B)
            losses_SURE_NPMLE = losses_SURE_NPMLE[-1]
            losses_NLL_NPMLE = losses_NLL_NPMLE[-1]
            # print(f"NPMLE SURE: {losses_SURE_NPMLE[-1]/n}")
            # print(f"NPMLE NLL: {losses_NLL_NPMLE[-1]/n}")
            # print(f"NPMLE MSE: {np.linalg.norm(theta_hats_NPMLE[-1,:] - theta)**2 / n}

            rhos.append(rho_Ztheta)
            BayesRisk.append(bayes_risk)
            empirical_BayesRisk.append(empirical_bayesrisk)
            MLE_MSE.append(np.linalg.norm(theta - Z.detach().numpy())**2 / n)

            EB_SURE.append(losses_SURE_EB)
            EB_NLL.append(losses_NLL_EB)
            EB_MSE.append(two_norm_differences_EB / n)

            NPMLE_SURE.append(losses_SURE_NPMLE)
            NPMLE_NLL.append(losses_NLL_NPMLE)
            NPMLE_MSE.append(two_norm_differences_NPMLE / n)
    
    for new_key in ['rhos', 'BayesRisk', 'empirical_BayesRisk', 'MLE_MSE',
                    'EB_SURE', 'EB_NLL', 'EB_MSE',
                    'NPMLE_SURE', 'NPMLE_NLL', 'NPMLE_MSE']:
        
        problem[new_key] = eval(new_key)

    problem.pop('B_dmplus1', None)

    return pd.DataFrame(problem)

def make_df(problems, m_sim=50, drop_sigma=False):

    results = [] # list of dataframes

    for problem in problems:        
        for m in range(m_sim):
            print(f"m_sim: {m}")
            problem_copy = problem.copy() # shallow copy is fine
            results.append(get_simulation_result(problem_copy, drop_sigma=drop_sigma))

    results_df = pd.concat(results)

    return results_df

