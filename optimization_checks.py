import simulate_data
import train
import models
import torch as tr
import pandas as pd
import numpy.random as rn

def simulate_location_scale_with_seed(random_seed,
                            ns=[100, 200, 400, 800, 1600, 3200],
                          experiments=["c", "d", "e"], B=100,
                          optimizer_str="adam"):
    '''
    Function that gives results for the eight different experiment simulations with different models. This is the 
    final function used to get results of figure 1 in section 6.2. This also takes into account if location and 
    scale will be used in our model. 
    '''

    experiments_list = []
    ns_list = []
    seed_list = []

    train_MSE_misspec_list = []
    train_SURE_misspec_list = []

    train_MSE_NPMLE_list = [] 
    train_SURE_NPMLE_list = [] 

    train_MSE_NPMLEinit_list = [] 
    train_SURE_NPMLEinit_list = [] 

    is_NPMLE_SURE_lower_than_SURE_PM_list = []

    for n in ns:

        print(f"n: {n}")

        for experiment in experiments:

            print(f"experiment: {experiment}")

            rn.seed(random_seed)

            theta, Z, X = simulate_data.xie(experiment=experiment, n=n)

            # Train methods that don't depend on scale or location

            found_NPMLE_solution=False

            try: 

                # NPMLE - misspecified. 
                # CPU 
                result_NPMLE = train.train_npmle(n, B, Z, theta, X) 
                problem_NPMLE, loss_NPMLE, score_NPMLE, theta_hat_NPMLE, twonorm_diff_NPMLE, pi_hat_NPMLE = result_NPMLE
                pi_hat_NPMLE[pi_hat_NPMLE < 0] = 0
                model_NPMLE = models.model_pi_sure(Z=Z, B=B, init_val=tr.log(pi_hat_NPMLE), device="cpu") # to compute SURE, theta hat
                SURE_NPMLE = model_NPMLE.opt_func(Z.cpu(), n, B, sigma=X[:,-1].cpu()).item()
                found_NPMLE_solution = True

            except Exception as e:
                print(f"Mosek failed on this run") 


            ns_list.append(n)
            experiments_list.append(experiment)
            seed_list.append(random_seed)

            if found_NPMLE_solution:
                # EB - NPMLEinit
                result_NPMLEinit = train.train_no_covariates(n, B, Z, theta, X, opt_objective = 'both', init_val_pi = tr.log(pi_hat_NPMLE),
                                                            optimizer_str=optimizer_str, use_location=True, use_scale=True, device=simulate_data.device) 
                model_NPMLEinit, SURE_NPMLEinit, score_NPMLEinit, theta_hats_NPMLEinit, twonorm_diff_NPMLEinit = result_NPMLEinit
                SURE_NPMLEinit = SURE_NPMLEinit[-1]
                print(f"Finished training EB misspecified with NPMLE init, with SURE: {SURE_NPMLEinit}")
                print(f"Finished training EB misspecified with NPMLE init, with in-sample MSE: {twonorm_diff_NPMLEinit / n}")

            # EB - misspecified
            result_misspec = train.train_no_covariates(n, B, Z, theta, X, opt_objective = 'both',
                                                    optimizer_str=optimizer_str, use_scale=True, use_location=True, device=simulate_data.device) 
            model_misspec, SURE_misspec, score_misspec, theta_hats_misspec, twonorm_diff_misspec = result_misspec
            SURE_misspec = SURE_misspec[-1]
            print(f"Finished training EB misspecified, with SURE: {SURE_misspec}")
            print(f"Finished training EB misspecified, with in-sample MSE: {twonorm_diff_misspec / n}")

            print("\n Training SURE of the models:")
            print(f"SURE-PM: {SURE_misspec}")
            print(f"NPMLE: {SURE_NPMLE}")
            print(f"SURE-PM, NPMLE init: {SURE_NPMLEinit}")

            if SURE_misspec > SURE_NPMLEinit:
                print(f"NPMLE's SURE is lower by {SURE_misspec - SURE_NPMLEinit}.")
                print(f"Seed: {random_seed}")
                is_NPMLE_SURE_lower_than_SURE_PM_list.append(True)
            else:
                is_NPMLE_SURE_lower_than_SURE_PM_list.append(False)

            # Append train results
            train_MSE_misspec_list.append(twonorm_diff_misspec/n)
            train_SURE_misspec_list.append(SURE_misspec)

            if found_NPMLE_solution:
                train_MSE_NPMLE_list.append(twonorm_diff_NPMLE/n)
                train_MSE_NPMLEinit_list.append(twonorm_diff_NPMLEinit/n)
                train_SURE_NPMLE_list.append(SURE_NPMLE)
                train_SURE_NPMLEinit_list.append(SURE_NPMLEinit)
            else:
                train_MSE_NPMLE_list.append(None)
                train_MSE_NPMLEinit_list.append(None)
                train_SURE_NPMLE_list.append(None)
                train_SURE_NPMLEinit_list.append(None)

    mse_sure_df = pd.DataFrame({'n': ns_list,
                                'experiment': experiments_list,
                                'MSE_misspec': train_MSE_misspec_list ,
                                'MSE_NPMLEinit': train_MSE_NPMLEinit_list,
                                'MSE_NPMLE': train_MSE_NPMLE_list,
                                'SURE_misspec': train_SURE_misspec_list, 
                                'SURE_NPMLEinit': train_SURE_NPMLEinit_list,
                                'SURE_NPMLE': train_SURE_NPMLE_list,
                                'data': len(ns)*len(experiments)*['train'],
                                'NPMLE_lower_SURE': is_NPMLE_SURE_lower_than_SURE_PM_list,
                                'seed': seed_list}) 
    
    return(mse_sure_df)

def make_df(ns=[100, 200, 400, 800, 1600, 3200],
            experiments=["c", "d", "e"],
            optimizer_str="adam", m_sim=1000, B=100):
    """
    Compute MSEs and SURE on train and test for all models, m_sim times and returns a concatenated dataframe. 
    """

    mse_sure_results = [] # list of dataframes
    
    for m in range(m_sim):
        print(f"m_sim: {m}")
        mse_sure_results.append(simulate_location_scale_with_seed(random_seed=m_sim,
                                                                  ns=ns, experiments=experiments, B=B,
                                                        optimizer_str=optimizer_str)) 

    mse_sure_df = pd.concat(mse_sure_results)

    return mse_sure_df 
