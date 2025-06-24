import simulate_data
import train
import models
import numpy as np 
import pandas as pd
import torch as tr
import numpy.random as rn

####################### score for covariates #########################

def normal_score_simulation(ns=[100, 500, 1000, 2000, 10000],
                            mu_theta=10, sigma_theta=1, sigma=1, B=100):
    """
    theta ~ N(10, 1)
    Z | theta ~ N(theta, 1)

    Scores for this setting using models:
    * NPMLE
    * EB, no covariates code
    * EB, covariates code (that takes a constant as the covariate)

    Saves csv 
    """

    df_score_list = []

    for n in ns: 
        print(f"n: {n}")

        count = 0
        count_exceptions = 0  # Counts for when MOSEK fails

        while count < 1:

            try: 
                # Simulate train data
                theta, Z, X = simulate_data.simulate_data_normal_nocovariates(n, mu_theta, sigma_theta, sigma) 
                Z_grid = np.linspace(min(Z), max(Z), n)
                Z_grid = tr.tensor(Z_grid, requires_grad=False)

                # Solve for NPMLE solution
                results_NPMLE = train.train_npmle(n, B, Z, theta, X) 
                problem_NPMLE, loss_NPMLE, score_NPMLE, theta_hat_NPMLE, twonorm_diff_NPMLE, pi_hat_NPMLE = results_NPMLE
                pi_hat_NPMLE[pi_hat_NPMLE < 0] = 0
                print("Finished solving NPMLE solution")

                count += 1

            except Exception as e:
                count_exceptions += 1
                print(f"MOSEK failed.")

                if (n == 100000 or n == 50000) and count_exceptions == 5:
                    count += 1


        print(f"For n={n}, MOSEK failed {count_exceptions} times.")

        # Train model: no covariates, EB (both theta and pi)
        results_nocovariates = train.train_no_covariates(n, B, Z, theta, X, opt_objective = 'both')
        model_nocovariates, SURE_nocovariates, score_nocovariates, theta_hats_nocovariates, twonorm_diff_nocovariates = results_nocovariates 
        print("Finished training model: no covariates")

        # Train model: covariates, EB
        X = tr.cat((X, X), 1) # X has to be n x 2 for train_covariates
        results_covariates = train.train_covariates(X, Z, theta, objective="SURE", B=B, d=2)
        model_covariates, Ft_Rep_covariates, SURE_covariates_cov, NLL_covariates, score_covariates, theta_hats_covariates, two_norm_differences_covariates = results_covariates
        print("Finished training model: covariates")

        # Compute scores
        score_nocovariates_grid = model_nocovariates.compute_score(Z_grid, n, B, sigma=X[:, -1] ).detach().numpy()
        score_covariates_grid = model_covariates.compute_score(Z_grid, Ft_Rep_covariates, X).detach().numpy()

        # print(f"score_nocovariates_grid: {score_nocovariates_grid}")
        # print(f"score_covariates_grid: {score_covariates_grid}")
        # print("###")
        
        # scores from NPMLE solution
        if count_exceptions < 5:
            model_NPMLEinit =  models.model_pi_sure(B, init_val=tr.log(pi_hat_NPMLE)) 
            score_NPMLE_grid = model_NPMLEinit.compute_score(Z_grid, n, B, sigma=X[:, -1]).detach().numpy()

        if count_exceptions < 5:
            df_score = pd.DataFrame({'nocovariates': score_nocovariates_grid,
                                    'covariates': score_covariates_grid,
                                    'NPMLE': score_NPMLE_grid, # NPMLE doesn't have train vs. test score
                                    'Z': Z.detach().numpy(), # NPMLE scores are only on train data
                                    'Z_grid': Z_grid.detach().numpy(),
                                    'n': n*[n]})
        
        else:
            df_score = pd.DataFrame({'nocovariates': score_nocovariates_grid,
                                    'covariates': score_covariates_grid,
                                    'NPMLE': n*[0], 
                                    'Z': Z.detach().numpy(), # NPMLE scores are only on train data
                                    'Z_grid': Z_grid.detach().numpy(),
                                    'n': n*[n]})
            
                
        df_score_list.append(df_score)

    df_score_final = pd.concat(df_score_list, axis=0, ignore_index=True)
    df_score_final.to_csv('results/score_normal.csv')

def xie_score_simulation(ns = [100, 1000], B = 100,
                         experiments = ["c", "d", "e", "f"],
                         variance_dict = {"c": [0.2, 0.55, 0.9],
                                        "d": [0.01, 0.125, 1],
                                        "e": [0.1, 0.5], 
                                        "f": [0.2, 0.55, 0.9]}):
    """
    In experiments c) through f), the data are heteroskedastic
    and the data's mean and variance are the same (theta = A). 
    
    The NPMLE is misspecified because the objective minimizes the KL divergence. 
    
    Both EB methods minimizes MSE. EB with covariates is well-specified;
    EB without covariates is misspecified (because it doesn't have
    A in the training).

    We think EB without covariates is still better specified than NPMLE.

    When Mosek fails more than 5x, skip methods with NPMLE for that iteration and
    append -100 as the SURE and MSE.
    """
    # covariates, no covariates (misspecified)
    # NPMLE, EB

    experiments_list = []
    ns_list = []
    df_score_list = []

    for n in ns:

        print(f"n: {n}")

        for experiment in experiments:

            ### Train estimators ###
            print(f"Training experiment: {experiment}")

            experiments_list.append(experiment)
            ns_list.append(n)

            count = 0
            count_exceptions = 0
            while count < 1:
                try: 
                    theta, Z, X = simulate_data.xie(experiment=experiment, n=n)

                    # Will calculate score at these points
                    Z_grid = np.linspace(min(Z), max(Z), n)
                    Z_grid = tr.tensor(Z_grid, requires_grad=False)

                    # NPMLE - misspecified
                    result_NPMLE = train.train_npmle(n, B, Z, theta, X) 
                    problem_NPMLE, loss_NPMLE, score_NPMLE, theta_hat_NPMLE, twonorm_diff_NPMLE, pi_hat_NPMLE = result_NPMLE
                    pi_hat_NPMLE[pi_hat_NPMLE < 0] = 0

                    count = count + 1
                except Exception as e:

                    count_exceptions = count_exceptions + 1
                    print(f"Mosek failed.")

                    if count_exceptions > 5:
                        count += 1 
            
            print(f"Mosek failed {count_exceptions} number of times.")

            ### Training 

            # EB - misspecified
            result_misspec = train.train_no_covariates(n, B, Z, theta, X, opt_objective = 'both') 
            model_misspec, SURE_misspec, score_misspec, theta_hats_misspec, twonorm_diff_misspec = result_misspec

            # EB - misspecified with NPMLEinit
            if count_exceptions < 5:
                result_NPMLEinit = train.train_no_covariates(n, B, Z, theta, X, opt_objective = 'both', init_val_pi = tr.log(pi_hat_NPMLE)) 
                model_NPMLEinit, SURE_NPMLEinit, score_NPMLEinit, theta_hats_NPMLEinit, twonorm_diff_NPMLEinit = result_NPMLEinit

            # EB - wellspecified
            results_wellspec = train.train_covariates(X, Z, theta)
            model_wellspec, Ft_Rep_wellspec, SURE_wellspec, NLL_wellspec, score_wellspec, theta_hats_wellspec, twonorm_diff_wellspec = results_wellspec

            ### Scores for difference values of sigma ###
            
            variance_list = variance_dict[experiment]

            for variance_i in variance_list:

                sigma_i = np.sqrt(variance_i)
                Z_grid = simulate_data.xie_Z_grid(n, experiment, sigma_i)
                # print(f"Z_grid: {Z_grid}")
                # print(f"sigma_i: {sigma_i}")
                sigma_tensor = sigma_i*tr.ones(n,1)

                # NPMLE
                if count_exceptions < 5:
                    model_NPMLE =  models.model_pi_sure(Z, B, init_val=tr.log(pi_hat_NPMLE)) # Z is train
                    score_NPMLE_grid = model_NPMLE.compute_score(Z_grid, n, B, sigma=sigma_tensor.reshape(n,),).detach().numpy()
                    # print(f"score_NPMLE_grid: {score_NPMLE_grid.shape}")


                    # EB - misspecified with NPMLE initialization 
                    score_NPMLEinit_grid = model_NPMLEinit.compute_score(Z_grid, n, B, sigma_tensor.reshape(n,)).detach().numpy()

                # EB - misspecified
                score_misspec_grid = model_misspec.compute_score(Z_grid, n, B, sigma_tensor.reshape(n,)).detach().numpy()
                # Returns n x n matrix where each row is the same, so take the first row

                # EB - well specified
                Ft_Rep = model_wellspec.feature_representation(sigma_tensor)
                # print(f"Ft_Rep: {Ft_Rep}")
                score_wellspec_grid = model_wellspec.compute_score(Z_grid, Ft_Rep, sigma_tensor).detach().numpy()

                # print(f"score_NPMLE_grid.shape: { score_NPMLE_grid.shape}")
                # print(f"score_NPMLEinit_grid.shape: { score_NPMLEinit_grid.shape}")
                # print(f"score_misspec_grid.shape: { score_misspec_grid.shape}")
                # print(f"score_wellspec_grid.shape: { score_wellspec_grid.shape}")
                # print(f"Z.detach().numpy().shape: { Z.detach().numpy().shape}")
                # print(f"Z_grid.detach().numpy().shape: { Z_grid.detach().numpy().shape}")
                # print(f"X.detach().numpy().reshape(n,).shape: { X.detach().numpy().reshape(n,).shape}")

                if count_exceptions < 5:
                    df_score_list.append(pd.DataFrame(
                        {'score_NPMLE': score_NPMLE_grid, 
                        'score_NPMLEinit': score_NPMLEinit_grid,
                        'score_misspec': score_misspec_grid,
                        'score_wellspec': score_wellspec_grid,
                        'sigma': n*[sigma_i],
                        'Z': Z.detach().numpy(),
                        'Z_grid': Z_grid.detach().numpy(),
                        'n': n*[n],
                        'X': X.detach().numpy().reshape(n,),
                        'experiment': n*[experiment]}))
                else:
                    df_score_list.append(pd.DataFrame(
                        {'score_NPMLE': n*[0], # NPMLE wasn't trained
                        'score_NPMLEinit': n*[0], # NPMLEinit wasn't trained
                        'score_misspec': score_misspec_grid,
                        'score_wellspec': score_wellspec_grid,
                        'sigma': n*[sigma_i],
                        'Z': Z.detach().numpy(),
                        'Z_grid': Z_grid.detach().numpy(),
                        'n': n*[n],
                        'X': X.detach().numpy().reshape(n,),
                        'experiment': n*[experiment]}))


    df_score_final = pd.concat(df_score_list, axis=0, ignore_index=True)
    df_score_final.to_csv("results/score_xie.csv")


# normal_score_simulation(ns = [100, 500, 1000])

xie_score_simulation(ns = [500, 1000, 5000])

####################### score for no covariates #########################

def nocovariates_score_simulation(experiment = 'Normal', 
                                  Z_dist = 'Heteroskedastic', 
                                  ns = [1000, 5000, 10000], 
                                  sigma_list = [0.55, 0.75, 1, 1.25, 1.45], 
                                  sigma_thetas = [0.1, 1, 5], 
                                  val_thetas = [3, 5, 7], 
                                  ks = [5, 50, 500], 
                                  mu_theta = 10, 
                                  B = 100): 
    """
    theta ~ N(10, sigma_theta) or k of them equals val_theta and 0 otherwise 
    Z | theta ~ N(theta, 1) 

    Scores for this setting using models:
    * NPMLE
    * EB, no covariates code 
    * True 

    Saves csv 
    """
    df_score_list = []

    if experiment == 'Normal': 

        for n in ns:

            print(f"n: {n}")

            if Z_dist == 'Homoscedastic': 
                sigma = np.ones(n) # Homoscedastic
            elif Z_dist == 'Heteroskedastic': 
                sigma = rn.uniform(0.5, 1.5, size=(n,)) # Heteroscedastic 

            for sigma_theta in sigma_thetas: 

                count = 0

                while count < 1: 

                    try: 
                        # Simulate data
                        theta, Z, X = simulate_data.simulate_data_normal_nocovariates(n, mu_theta, sigma_theta, sigma)
                        Z_grid = tr.tensor(np.linspace(min(Z), max(Z), n), requires_grad=False)

                        # NPMLE 
                        results_NPMLE = train.train_npmle(n, B, Z, theta, X) 
                        problem_NPMLE, loss_NPMLE, score_NPMLE, theta_hat_NPMLE, twonorm_diff_NPMLE, pi_hat_NPMLE = results_NPMLE 
                        pi_hat_NPMLE[pi_hat_NPMLE < 0] = 0

                        count += 1

                    except Exception as e:
                        print(f"Mosek failed.") 
                
                # Score for NPMLE
                model = models.model_pi_sure(Z, B, init_val=tr.log(pi_hat_NPMLE)) 
                score_NPMLE = model.compute_score(Z, n, B, X[:,-1]).detach().numpy() 

                # SURE fitting 
                results_EB_both = train.train_no_covariates(n, B, Z, theta, X, opt_objective = 'both')
                model_EB_both, SURE_EB_both, scores_EB_both, theta_hats_EB_both, twonorm_diff_EB_both = results_EB_both 
                score_SURE = model_EB_both.compute_score(Z, n, B, X[:,-1]).detach().numpy() 

                if Z_dist == 'Homoscedastic': 

                    score_grid_homo_NPMLE =  model.compute_score(Z_grid, n, B, sigma=tr.ones(n))
                    theta_hat_grid_homo_NPMLE = Z_grid + (tr.ones(n)**2) * score_grid_homo_NPMLE
                    score_grid_homo_SURE =  model_EB_both.compute_score(Z_grid, n, B, sigma=tr.ones(n)) 
                    theta_hat_grid_homo_SURE = Z_grid + (tr.ones(n)**2) * score_grid_homo_SURE
                    score_grid_homo_truth = (mu_theta - Z_grid) / (sigma**2 + sigma_theta**2)
                    theta_hat_grid_homo_truth = Z_grid + (tr.ones(n)**2) * score_grid_homo_truth

                    df_score = pd.DataFrame({'n': n*np.ones(n), 
                                             'sigma_theta': sigma_theta*np.ones(n), 
                                             'Z': Z.detach().numpy(), 
                                             'Z_grid': Z_grid.detach().numpy(),
                                             'NPMLE': score_NPMLE, 
                                             'SURE': score_SURE, 
                                             'NPMLE_grid': score_grid_homo_NPMLE.detach().numpy(), 
                                             'theta_hat_NPMLE_grid': theta_hat_grid_homo_NPMLE.detach().numpy(), 
                                             'SURE_grid': score_grid_homo_SURE.detach().numpy(), 
                                             'theta_hat_SURE_grid': theta_hat_grid_homo_SURE.detach().numpy(), 
                                             'TRUTH_grid': score_grid_homo_truth.detach().numpy(), 
                                             'theta_hat_TRUTH_grid': theta_hat_grid_homo_truth.detach().numpy()}) 
                    
                    df_score_list.append(df_score) 

                if Z_dist == 'Heteroskedastic': 

                    for sigma_i in sigma_list: 

                        score_grid_hetero_NPMLE =  model.compute_score(Z_grid, n, B, sigma=sigma_i*tr.ones(n)) 
                        theta_hat_grid_hetero_NPMLE = Z_grid + (sigma_i**2) * score_grid_hetero_NPMLE
                        score_grid_hetero_SURE =  model_EB_both.compute_score(Z_grid, n, B, sigma=sigma_i*tr.ones(n))
                        theta_hat_grid_hetero_SURE = Z_grid + (sigma_i**2) * score_grid_hetero_SURE
                        score_grid_hetero_truth = (mu_theta - Z_grid) / (sigma_i**2 + sigma_theta**2) 
                        theta_hat_grid_hetero_truth = Z_grid + (sigma_i**2) * score_grid_hetero_truth

                        df_score = pd.DataFrame({'n': n*np.ones(n), 
                                                 'sigma_i': sigma_i*np.ones(n), 
                                                 'sigma_theta': sigma_theta*np.ones(n), 
                                                 'Z': Z.detach().numpy(), 
                                                 'Z_grid': Z_grid.detach().numpy(),
                                                 'NPMLE': score_NPMLE, 
                                                 'SURE': score_SURE, 
                                                 'NPMLE_grid': score_grid_hetero_NPMLE.detach().numpy(), 
                                                 'theta_hat_NPMLE_grid': theta_hat_grid_hetero_NPMLE.detach().numpy(), 
                                                 'SURE_grid': score_grid_hetero_SURE.detach().numpy(), 
                                                 'theta_hat_SURE_grid': theta_hat_grid_hetero_SURE.detach().numpy(), 
                                                 'TRUTH_grid': score_grid_hetero_truth.detach().numpy(), 
                                                 'theta_hat_TRUTH_grid': theta_hat_grid_hetero_truth.detach().numpy()}) 
                    
                        df_score_list.append(df_score) 
                
                print("Finished computing scores") 
    
    elif experiment == 'Binary': 

        for n in ns:

            print(f"n: {n}")

            if Z_dist == 'Homoscedastic': 
                sigma = np.ones(n) # Homoscedastic
            elif Z_dist == 'Heteroskedastic': 
                sigma = rn.uniform(0.5, 1.5, size=(n,)) # Heteroscedastic 

            for val_theta in val_thetas: 

                for k in ks: 

                    count = 0

                    while count < 1: 

                        try: 
                            # Simulate data
                            theta, Z, X = simulate_data.simulate_data_discrete_nocovariates(n, k, val_theta, sigma) 
                            Z_grid = tr.tensor(np.linspace(min(Z), max(Z), n), requires_grad=False)

                            # NPMLE 
                            results_NPMLE = train.train_npmle(n, B, Z, theta, X) 
                            problem_NPMLE, loss_NPMLE, score_NPMLE, theta_hat_NPMLE, twonorm_diff_NPMLE, pi_hat_NPMLE = results_NPMLE 
                            pi_hat_NPMLE[pi_hat_NPMLE < 0] = 0

                            count += 1

                        except Exception as e:
                            print(f"Mosek failed.") 
                    
                    # Score for NPMLE
                    model = models.model_pi_sure(Z, B, init_val=tr.log(pi_hat_NPMLE)) 
                    score_NPMLE = model.compute_score(Z, n, B, X[:,-1]).detach().numpy() 

                    # SURE fitting 
                    results_EB_both = train.train_no_covariates(n, B, Z, theta, X, opt_objective = 'both')
                    model_EB_both, SURE_EB_both, scores_EB_both, theta_hats_EB_both, twonorm_diff_EB_both = results_EB_both 
                    score_SURE = model_EB_both.compute_score(Z, n, B, X[:,-1]).detach().numpy() 

                    if Z_dist == 'Homoscedastic': 

                        score_grid_homo_NPMLE =  model.compute_score(Z_grid, n, B, sigma=tr.ones(n))
                        theta_hat_grid_homo_NPMLE = Z_grid + (tr.ones(n)**2) * score_grid_homo_NPMLE
                        score_grid_homo_SURE =  model_EB_both.compute_score(Z_grid, n, B, sigma=tr.ones(n)) 
                        theta_hat_grid_homo_SURE = Z_grid + (tr.ones(n)**2) * score_grid_homo_SURE
                        score_grid_homo_truth = ((k/n)*((val_theta-Z_grid)/(tr.ones(n)**2))*tr.exp(-0.5*(((Z-val_theta)/tr.ones(n))**2)) 
                                                 + (1-k/n)*((-Z_grid)/(tr.ones(n)**2))*tr.exp(-0.5*((Z/tr.ones(n))**2))) / ((k/n)*tr.exp(-0.5*(((Z-val_theta)/tr.ones(n))**2)) 
                                                                                                                       + (1-k/n)*tr.exp(-0.5*((Z/tr.ones(n))**2)))
                        theta_hat_grid_homo_truth = Z_grid + (tr.ones(n)**2) * score_grid_homo_truth

                        df_score = pd.DataFrame({'n': n*np.ones(n), 
                                                'k': k*np.ones(n), 
                                                'mu': val_theta*np.ones(n), 
                                                'Z': Z.detach().numpy(), 
                                                'Z_grid': Z_grid.detach().numpy(),
                                                'NPMLE': score_NPMLE, 
                                                'SURE': score_SURE, 
                                                'NPMLE_grid': score_grid_homo_NPMLE.detach().numpy(), 
                                                'theta_hat_NPMLE_grid': theta_hat_grid_homo_NPMLE.detach().numpy(), 
                                                'SURE_grid': score_grid_homo_SURE.detach().numpy(), 
                                                'theta_hat_SURE_grid': theta_hat_grid_homo_SURE.detach().numpy(), 
                                                'TRUTH_grid': score_grid_homo_truth.detach().numpy(), 
                                                'theta_hat_TRUTH_grid': theta_hat_grid_homo_truth.detach().numpy()}) 
                        
                        df_score_list.append(df_score) 

                    if Z_dist == 'Heteroskedastic': 

                        for sigma_i in sigma_list: 

                            score_grid_hetero_NPMLE =  model.compute_score(Z_grid, n, B, sigma=sigma_i*tr.ones(n)) 
                            theta_hat_grid_hetero_NPMLE = Z_grid + (sigma_i**2) * score_grid_hetero_NPMLE
                            score_grid_hetero_SURE =  model_EB_both.compute_score(Z_grid, n, B, sigma=sigma_i*tr.ones(n))
                            theta_hat_grid_hetero_SURE = Z_grid + (sigma_i**2) * score_grid_hetero_SURE
                            score_grid_hetero_truth = ((k/n)*((val_theta-Z_grid)/(sigma_i**2))*tr.exp(-0.5*(((Z-val_theta)/sigma_i)**2)) 
                                                       + (1-k/n)*((-Z_grid)/(sigma_i**2))*tr.exp(-0.5*((Z/sigma_i)**2))) / ((k/n)*tr.exp(-0.5*(((Z-val_theta)/sigma_i)**2)) 
                                                                                                                            + (1-k/n)*tr.exp(-0.5*((Z/sigma_i)**2))) 
                            theta_hat_grid_hetero_truth = Z_grid + (sigma_i**2) * score_grid_hetero_truth

                            df_score = pd.DataFrame({'n': n*np.ones(n), 
                                                    'sigma_i': sigma_i*np.ones(n), 
                                                    'k': k*np.ones(n), 
                                                    'mu': val_theta*np.ones(n), 
                                                    'Z': Z.detach().numpy(), 
                                                    'Z_grid': Z_grid.detach().numpy(),
                                                    'NPMLE': score_NPMLE, 
                                                    'SURE': score_SURE, 
                                                    'NPMLE_grid': score_grid_hetero_NPMLE.detach().numpy(), 
                                                    'theta_hat_NPMLE_grid': theta_hat_grid_hetero_NPMLE.detach().numpy(), 
                                                    'SURE_grid': score_grid_hetero_SURE.detach().numpy(), 
                                                    'theta_hat_SURE_grid': theta_hat_grid_hetero_SURE.detach().numpy(), 
                                                    'TRUTH_grid': score_grid_hetero_truth.detach().numpy(), 
                                                    'theta_hat_TRUTH_grid': theta_hat_grid_hetero_truth.detach().numpy()}) 
                        
                            df_score_list.append(df_score) 

                    print("Finished computing scores") 

    df_score_final = pd.concat(df_score_list, axis=0, ignore_index=True) 

    return df_score_final 

nocovariates_score_simulation(experiment = 'Normal', Z_dist = 'Heteroskedastic').to_csv('results/score_nocovariates_normal_hetero.csv') 
# nocovariates_score_simulation(experiment = 'Binary', Z_dist = 'Heteroskedastic').to_csv('results/score_nocovariates_binary_hetero.csv') 
nocovariates_score_simulation(experiment = 'Normal', Z_dist = 'Homoscedastic').to_csv('results/score_nocovariates_normal_homo.csv') 
# nocovariates_score_simulation(experiment = 'Binary', Z_dist = 'Homoscedastic').to_csv('results/score_nocovariates_binary_homo.csv') 