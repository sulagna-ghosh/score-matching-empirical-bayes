"""
ATLAS data fission experiment *without CLOSE-NPMLE*. (Running CLOSE experiments locally in Jupyter Notebook, not on cluster)
"""

import sys  
import os

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


import pandas as pd 
import numpy as np
import torch as tr
import scipy.stats as ss
from simulate_data import device

# Our EB methods
from train import train_no_covariates as train_no_covariates
from train import train_npmle as train_npmle
from train import train_covariates as train_covariates
from train import train_sure_ls as train_sure_ls

import cvxpy as cp 
import math

import pyfixest as pf



def load_data_for_outcome(est_var, input_dir="atlas_data/oa_data_used.feather"):
    """
    Load the processed data for a given outcome variable
    Filter out missing values and values where the standard error is too large
    """
    df = pd.read_feather(input_dir)
    se_var = est_var + "_se"
    subset = df[[est_var, se_var, "czname", "state", "county", "tract"] + clean_covariates].dropna()
    thresh = subset[se_var].quantile(0.995)
    subset = subset.loc[subset[se_var] <= thresh].reset_index(drop=True)

    return subset

clean_covariates = ['par_rank_pooled_pooled_mean',
 'par_rank_black_pooled_mean',
 'poor_share2010',
 'share_black2010',
 'hhinc_mean2000',
 'ln_wage_growth_hs_grad',
 'frac_coll_plus2010',
 'log_kid_black_pooled_blw_p50_n',
 'log_kid_pooled_pooled_blw_p50_n']



est_var = "kfr_top20_black_pooled_p25"
df = load_data_for_outcome(est_var)
estimates = df[est_var].values
sigma_np = df[est_var + "_se"].values

n = df.shape[0]
B = 100
Z = tr.tensor(estimates).to(device)
X_sigma = tr.tensor(sigma_np.reshape(n, 1)).to(device)

use_location = False
use_scale = True

m_sim = 30
alpha=1

np.random.seed(5012025)
epsilon_matrix = np.concatenate([ss.norm.rvs(loc=0, scale=sigma_np).reshape(n, 1) for i in range(m_sim)], axis = 1)
epsilon_matrix_for_fixing_Z = np.concatenate([ss.norm.rvs(loc=0, scale=sigma_np).reshape(n, 1) for i in range(m_sim)], axis = 1)
np.save("miscellaneous/epsilon_matrix_5012025.npy", epsilon_matrix)

# Functions ###

def residualize(df, est_var, covariates, weighted=True, czvar="czname", within_cz=False):
    """Residualize `est_var` on `covariates`, optionally within CZs."""
    df = df.copy()
    se_var = est_var + "_se"

    if weighted:
        df["w"] = 1 / (df[se_var] ** 2)
    weights = "w" if weighted else None
    fmla = f"{est_var} ~ 1 + {' + '.join(covariates)}"

    if not within_cz:
        fit = pf.feols(fmla, weights=weights, data=df)
        residuals = fit.resid()
        fitted_values = fit.predict()
    else:
        # Residualize within each CZ
        residuals = np.full_like(df[est_var], np.nan)
        fitted_values = np.full_like(df[est_var], np.nan)
        for cz in df[czvar].unique():
            mask = df[czvar] == cz
            fit = pf.feols(fmla, weights=weights, data=df.loc[mask])
            residuals[mask] = fit.resid()
            fitted_values[mask] = fit.predict()

    return fitted_values, residuals


def train_verbose_npmle(n, B, Z, theta, X, accept_unknown=True): 


    if X.is_cuda | Z.is_cuda: 
        X = X.cpu()
        Z = Z.cpu()

    # sigma of Z
    sigma = X[:, -1] 

    theta_diff = (1/(B-1))*tr.ones(B-1) 
    theta_cum = tr.cumsum(theta_diff, dim = 0)*(max(Z)-min(Z))
    theta_cum = theta_cum[None, :]
    theta_cum = theta_cum.expand(n, B-1) 
    
    Z_nb = Z[:, None]
    Z_nb = Z_nb.expand(n, B) 
    Z_theta = Z_nb - min(Z)
    Z_theta[:, 1:] = Z_theta[:, 1:] - theta_cum
    Z_theta_sq = Z_theta**2
    Z_theta_by_sigma_sq = Z_theta_sq/(sigma[:,None]**2) 
    
    # (Re)Defining numpy variables
    Z_theta_by_sigma_sq_np = Z_theta_by_sigma_sq.detach().numpy() 
    sigma_np = sigma.detach().numpy() 
    norm_dens_mat = np.exp(-Z_theta_by_sigma_sq_np/2)/(((2*math.pi)**0.5)*sigma_np[:, None]) 

    # Variable to be optimized
    pi_np = cp.Variable(B)

    # Objective and constraints
    objective = cp.Maximize(cp.sum(cp.log(norm_dens_mat @ pi_np)))
    constraints = [cp.sum(pi_np) == 1, pi_np >= 0]

    # Optimization
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver = cp.MOSEK, verbose=True, accept_unknown=accept_unknown)

    # Optimized objective and varaible values
    loss = prob.value
    pi_hat = tr.tensor(pi_np.value)
    pi_hat[pi_hat < 0] = 0

    # Optmized theta_hat
    pi_param_est = pi_hat[None, :]
    pi_param_est = pi_param_est.expand(n, B)
    numerator = ((pi_param_est*(-1*Z_theta)*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1))/(sigma**2) 
    denominator = (pi_param_est*tr.exp(-Z_theta_by_sigma_sq/2)).sum(axis = 1)
    score = numerator/denominator 
    theta_hat = Z + sigma**2 *score

    # L2 norm
    two_norm_difference = (np.linalg.norm(theta_hat.detach().numpy() - theta)**2)

    return (prob, loss, score.detach().numpy(), theta_hat.detach().numpy(), two_norm_difference, pi_hat) 



def make_fissioned_datasets(m, Z=Z, alpha=alpha, epsilon_matrix=epsilon_matrix, sigma_np=sigma_np,
                            fixing_Z_for_debug=False):

    epsilon = epsilon_matrix[:, m]

    if (not fixing_Z_for_debug):

        Z_train = (Z + alpha*epsilon).cpu().detach().clone().to(device)
        scaled_sigma_train = tr.tensor(np.sqrt((1 + alpha**2) * sigma_np**2)).reshape(n,1).to(device)

        Z_evaluate = (Z - epsilon/alpha).cpu().detach().clone().to(device)
        scaled_sigma_evaluate = tr.tensor(np.sqrt((1 + 1/alpha**2) * sigma_np**2)).reshape(n,1).to(device)
    
    else: 

        epsilon_for_fixing_Z = epsilon_matrix_for_fixing_Z[:, m]
        new_Z = (Z + epsilon_for_fixing_Z).cpu().detach().clone().to(device)

        Z_train = (new_Z + alpha*epsilon).cpu().detach().clone().to(device)
        scaled_sigma_train = tr.tensor(np.sqrt((1 + alpha**2) * sigma_np**2)).reshape(n,1).to(device)

        Z_evaluate = (new_Z - epsilon/alpha).cpu().detach().clone().to(device)
        scaled_sigma_evaluate = tr.tensor(np.sqrt((1 + 1/alpha**2) * sigma_np**2)).reshape(n,1).to(device)

    return Z_train, scaled_sigma_train, Z_evaluate, scaled_sigma_evaluate

def fission_mse_without_close(m_sim=m_sim, Z=Z, alpha=alpha, epsilon_matrix=epsilon_matrix):

    mse_npmle_list = []
    mosek_failures_idx_list = []

    mse_pm_list = []
    # mse_thing_list = []
    # mse_ls_list = []
    # mse_thing_residualized_list = []
    # mse_thing_full_list = []
    # mse_thing_bivariate_list = []
    # mse_ls_residualized_list = []
    # mse_ls_full_list = []
    # mse_ls_bivariate_list = []
    mse_mle_list = []

    zeros_for_theta = np.zeros((n,))

    for m in range(m_sim):

        print(f"m: {m}")

        fissioned_datasets = make_fissioned_datasets(m, Z=Z, alpha=alpha, epsilon_matrix=epsilon_matrix, sigma_np=sigma_np)
        Z_train, scaled_sigma_train, Z_evaluate, scaled_sigma_evaluate = fissioned_datasets
        Z_evaluate_np = Z_evaluate.cpu().detach().numpy()

        # NPMLE 

        found_npmle_solution=False

        try: 

            result_npmle = train_npmle(n, B, Z_train, zeros_for_theta, scaled_sigma_train) 
            theta_hat_npmle_train = result_npmle[3]

            found_npmle_solution = True

            mse_npmle = np.sum((Z_evaluate_np - theta_hat_npmle_train)**2)/n

        except Exception as e:
            print(f"Mosek failed on the {m}-th run...") 

        
        if found_npmle_solution:
            mse_npmle_list.append(mse_npmle)
            mosek_failures_idx_list.append(np.nan)
        else:

            # accept unknown
            result_npmle = train_verbose_npmle(n, B, Z_train, zeros_for_theta, scaled_sigma_train, accept_unknown=True) 
            theta_hat_npmle_train = result_npmle[3]

            mse_npmle = np.sum((Z_evaluate_np - theta_hat_npmle_train)**2)/n
            
            mse_npmle_list.append(mse_npmle)
            mosek_failures_idx_list.append(m)
        
        # MLE
        mse_mle = np.sum((Z_evaluate_np - Z_train.cpu().detach().numpy())**2)/n
        mse_mle_list.append(mse_mle)

        # PM
        output_pm = train_no_covariates(n, B, Z_train, zeros_for_theta, 
                                  scaled_sigma_train, use_location=use_location, use_scale=use_scale, n_iter=2000)
        theta_hat_pm_train = output_pm[3][-1,:]
        mse_pm = np.sum((Z_evaluate_np - theta_hat_pm_train)**2)/n
        mse_pm_list.append(mse_pm)


    mse_df = pd.DataFrame(
        {'NPMLE': mse_npmle_list,
         'mosek_fail': ~np.isnan(mosek_failures_idx_list), 
         'MLE': mse_mle_list,
         'PM': mse_pm_list})

    return(mse_df)
