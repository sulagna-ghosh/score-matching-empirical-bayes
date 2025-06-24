import simulate_data
import train
import models

import numpy as np
import numpy.random as rn
import pandas as pd
import torch as tr 
import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
import random

# experiments c)-f) in xie 2012

m = 20

def get_simulation_result(ns = [100, 200, 400, 800, 1600, 3200, 6400],
                          experiments = ["c", "d", "d5", "e", "f"],
                          optimizer_str="adam", hidden_sizes=(8,8), B=100):
    # covariates, no covariates (misspecified)
    # NPMLE, EB

    experiments_list = []
    ns_list = []

    train_MSE_wellspec_list = [] # well specified EB
    train_MSE_misspec_list = [] # EB no covariates
    train_MSE_NPMLE_list = [] # NPMLE
    train_MSE_NPMLEinit_list = [] # EB no covariates with NPMLE initialization
    train_MSE_thetaG_list = [] # parametric baseline

    train_SURE_wellspec_list = []
    train_SURE_misspec_list = []
    train_SURE_NPMLE_list = []
    train_SURE_NPMLEinit_list = [] 
    train_SURE_thetaG_list = [] # this isn't really train

    test_MSE_wellspec_list = [] 
    test_MSE_misspec_list = [] 
    test_MSE_NPMLE_list = [] 
    test_MSE_NPMLEinit_list = [] 
    test_MSE_thetaG_list = [] # this isn't really test

    test_SURE_wellspec_list = []
    test_SURE_misspec_list = []
    test_SURE_NPMLE_list = []
    test_SURE_NPMLEinit_list = [] 
    test_SURE_thetaG_list = [] # this isn't really test



    for n in ns:

        print(f"n: {n}")

        for experiment in experiments:

            print(f"experiment: {experiment}")

            experiments_list.append(experiment)
            ns_list.append(n)

            count = 0
            count_exceptions = 0
                
            while count < 1:
                try: 
                    # Train data
                    theta, Z, X = simulate_data.xie(experiment=experiment, n=n)

                    # NPMLE - misspecified. 
                    result_NPMLE = train.train_npmle(n, B, Z, theta, X) 
                    problem_NPMLE, loss_NPMLE, score_NPMLE, theta_hat_NPMLE, twonorm_diff_NPMLE, pi_hat_NPMLE = result_NPMLE
                    pi_hat_NPMLE[pi_hat_NPMLE < 0] = 0
                    model_NPMLE = models.model_pi_sure(Z=Z, B=B, init_val=tr.log(pi_hat_NPMLE), device="cpu") # to compute SURE, theta hat
                    # model_NPMLE is CPU
                    print("Finished solving NPMLE")

                    # EB - misspecified (with NPMLE optmized pi as initial) 
                    result_NPMLEinit = train.train_no_covariates(n, B, Z, theta, X, opt_objective = 'both', init_val_pi = tr.log(pi_hat_NPMLE),
                                                                 optimizer_str=optimizer_str) 
                    model_NPMLEinit, SURE_NPMLEinit, score_NPMLEinit, theta_hats_NPMLEinit, twonorm_diff_NPMLEinit = result_NPMLEinit
                    SURE_NPMLEinit = SURE_NPMLEinit[-1]
                    print(f"Finished training EB misspecified with NPMLE init, with SURE: {SURE_NPMLEinit}")
                    count += 1

                except Exception as e:
                    count_exceptions +=  1
                    print(f"Error occurred") 

                    if count_exceptions == 5:
                        count += 1
            
            # EB - misspecified
            result_misspec = train.train_no_covariates(n, B, Z, theta, X, opt_objective = 'both',
                                                       optimizer_str=optimizer_str) 
            model_misspec, SURE_misspec, score_misspec, theta_hats_misspec, twonorm_diff_misspec = result_misspec
            SURE_misspec = SURE_misspec[-1]
            print(f"Finished training EB misspecified, with SURE: {SURE_misspec}")

            # EB - well specified
            result_wellspec = train.train_covariates(X, Z, theta, objective="SURE", set_seed=None, d=2, B=100, drop_sigma=False,
                                                       optimizer_str=optimizer_str, hidden_sizes=hidden_sizes)
            model_wellspec, Ft_Rep_wellspec, SURE_wellspec, NLL_wellspec, score_wellspec, theta_hats_wellspec, twonorm_diff_wellspec = result_wellspec
            SURE_wellspec = SURE_wellspec[-1]
            NLL_wellspec = NLL_wellspec[-1]
            print(f"Finished training EB wellspecified, with SURE: {SURE_wellspec}")
            

            # Xie - theta G parametric baseline
            theta_hat_G, MSE_thetaG, SURE_thetaG, grand_mean, lambda_hat = models.theta_hat_G(theta, Z, X)
            print(f"Finished solving parametric estimator, with SURE {SURE_thetaG}")
            
            if count_exceptions < 5:
                # NPMLE
                train_MSE_NPMLE_list.append(twonorm_diff_NPMLE/n)
                train_SURE_NPMLE_list.append(model_NPMLE.opt_func(Z.cpu(), n, B, sigma=X[:,-1].cpu()).item())
                
                # NPMLE init
                train_MSE_NPMLEinit_list.append(twonorm_diff_NPMLEinit/n) 
                train_SURE_NPMLEinit_list.append(SURE_NPMLEinit) 
            else:
                train_MSE_NPMLE_list.append(-100)
                train_SURE_NPMLE_list.append(-100)

                train_MSE_NPMLEinit_list.append(-100) 
                train_SURE_NPMLEinit_list.append(-100) 

            # misspec
            train_MSE_misspec_list.append(twonorm_diff_misspec/n)
            train_SURE_misspec_list.append(SURE_misspec)

            # wellspec
            train_MSE_wellspec_list.append(twonorm_diff_wellspec/n)
            train_SURE_wellspec_list.append(SURE_wellspec)

            # parametric
            train_MSE_thetaG_list.append(MSE_thetaG)
            train_SURE_thetaG_list.append(SURE_thetaG)

            ### Evaluate on test data ###
            theta, Z, X = simulate_data.xie(experiment=experiment, n=n)
            sigma = X[:,-1]

            if count_exceptions < 5: 
                # NPMLE evaluation
                # NPMLE is on CPU
                Z_cpu = Z.cpu()
                sigma_cpu = sigma.cpu()
                test_NPMLE_score = model_NPMLE.compute_score(Z_cpu, n, B, sigma_cpu) 
                test_NPMLE_thetahat = Z_cpu + sigma_cpu**2 * test_NPMLE_score
                test_MSE_NPMLE_list.append(np.linalg.norm(test_NPMLE_thetahat.detach().numpy() - theta)**2 / n)
                test_SURE_NPMLE_list.append(model_NPMLE.opt_func(Z_cpu, n, B, sigma_cpu).detach().numpy())

                # the rest of the models are GPU

                # NPMLEinit evaluation
                test_EB_NPMLEinit_score = model_NPMLEinit.compute_score(Z, n, B, sigma)
                test_EB_NPMLEinit_thetahat = Z + sigma**2 * test_EB_NPMLEinit_score
                test_MSE_NPMLEinit_list.append(np.linalg.norm(test_EB_NPMLEinit_thetahat.cpu().detach().numpy() - theta)**2 / n)
                test_SURE_NPMLEinit_list.append(model_NPMLEinit.opt_func(Z, n, B, sigma).cpu().detach().numpy())
            else:
                test_MSE_NPMLE_list.append(-100)
                test_SURE_NPMLE_list.append(-100)

                test_MSE_NPMLEinit_list.append(-100) 
                test_SURE_NPMLEinit_list.append(-100) 

            # EB misspec evaluation
            test_EB_misspec_score = model_misspec.compute_score(Z, n, B, sigma)
            test_EB_misspec_thetahat = Z + sigma**2 * test_EB_misspec_score
            test_MSE_misspec_list.append(np.linalg.norm(test_EB_misspec_thetahat.cpu().detach().numpy() - theta)**2 / n)
            test_SURE_misspec_list.append(model_misspec.opt_func(Z, n, B, sigma).cpu().detach().numpy())

            # EB wellspec evaluation
            test_Ft_Rep = model_wellspec.feature_representation(X)
            test_EB_wellspec_score = model_wellspec.compute_score(Z, test_Ft_Rep, X)
            test_EB_wellspec_thetahat = Z + sigma**2 * test_EB_wellspec_score
            test_MSE_wellspec_list.append(np.linalg.norm(test_EB_wellspec_thetahat.cpu().detach().numpy() - theta)**2 / n)
            test_SURE_wellspec_list.append(model_wellspec.opt_func_SURE(Z, test_Ft_Rep, X).cpu().detach().numpy())

            # parametric
            test_theta_hat_G, test_MSE_thetaG, test_SURE_thetaG, grand_mean, lambda_hat = models.theta_hat_G(theta, Z, X)
            test_MSE_thetaG_list.append(test_MSE_thetaG)
            test_SURE_thetaG_list.append(test_SURE_thetaG)

            print("Finished evaluation on test data")





    # print(f"experiment: {experiment}")
    # print(f"MSE_NPMLE_list: {MSE_NPMLE_list}")
    # print(f"MSE_wellspec_list: {MSE_wellspec_list}")
    # print(f"MSE_thetaG_list: {MSE_thetaG_list}")
    # print(f"SURE_thetaG_list: {SURE_thetaG_list}")



    mse_sure_df = pd.DataFrame({'n': 2*ns_list,
                              'experiment': 2*experiments_list,
                              'MSE_wellspec': train_MSE_wellspec_list + test_MSE_wellspec_list,
                              'MSE_misspec': train_MSE_misspec_list + test_MSE_misspec_list,
                              'MSE_NPMLEinit': train_MSE_NPMLEinit_list + test_MSE_NPMLEinit_list,
                              'MSE_NPMLE': train_MSE_NPMLE_list + test_MSE_NPMLE_list,
                              'MSE_thetaG': train_MSE_thetaG_list + test_MSE_thetaG_list,
                              'SURE_wellspec': train_SURE_wellspec_list + test_SURE_wellspec_list,
                              'SURE_misspec': train_SURE_misspec_list + test_SURE_misspec_list, 
                              'SURE_NPMLEinit': train_SURE_NPMLEinit_list + test_SURE_NPMLEinit_list,
                              'SURE_NPMLE': train_SURE_NPMLE_list + test_SURE_NPMLE_list,
                              'SURE_thetaG': train_SURE_thetaG_list + test_SURE_thetaG_list,
                              'data': len(experiments)*len(ns)*['train'] + len(experiments)*len(ns)*['test']}) 
    
        
    return(mse_sure_df)

def simulate_nn_sizes(ns=[100, 200, 400, 800, 1600, 3200, 6400],
                          experiments = ["c", "d", "d5", "e", "f"],
                          hidden_sizes_list=[(8,8), (256, 256), (256, 256, 256),
                                        (128, 128), (128, 128, 128)],
                            optimizer_str="adam", skip_connect=False):
    # covariates, no covariates (misspecified)
    # NPMLE, EB

    experiments_list = []
    ns_list = []
    hidden_sizes_simulation_list = []

    train_MSE_wellspec_list = [] # well specified EB
    train_SURE_wellspec_list = []
    
    test_MSE_wellspec_list = [] 
    test_SURE_wellspec_list = []

    for n in ns:

        print(f"n: {n}")

        for experiment in experiments:

            print(f"experiment: {experiment}")

            theta, Z, X = simulate_data.xie(experiment=experiment, n=n)
            theta_test, Z_test, X_test = simulate_data.xie(experiment=experiment, n=n)
            sigma_test = X_test[:,-1]

            for hidden_sizes in hidden_sizes_list:
            
                print(f"hidden_sizes: {hidden_sizes}")
                hidden_sizes_simulation_list.append(str(hidden_sizes))
                experiments_list.append(experiment)
                ns_list.append(n)

                # EB - well specified
                # GPU if available
                result_wellspec = train.train_covariates(X, Z, theta, objective="SURE", set_seed=None, d=2, B=100, drop_sigma=False,
                                                            optimizer_str=optimizer_str,
                                                            hidden_sizes=hidden_sizes, 
                                                            use_location=False, use_scale=True, 
                                                            device=simulate_data.device, skip_connect=skip_connect)
                model_wellspec, Ft_Rep_wellspec, SURE_wellspec, NLL_wellspec, score_wellspec, theta_hats_wellspec, twonorm_diff_wellspec = result_wellspec
                SURE_wellspec = SURE_wellspec[-1]
                NLL_wellspec = NLL_wellspec[-1]
                print(f"Finished training EB wellspecified, with SURE: {SURE_wellspec}")

                # wellspec
                train_MSE_wellspec_list.append(twonorm_diff_wellspec/n)
                train_SURE_wellspec_list.append(SURE_wellspec)

                # EB wellspec evaluation
                test_Ft_Rep = model_wellspec.feature_representation(X_test)
                test_EB_wellspec_score = model_wellspec.compute_score(Z_test, test_Ft_Rep, X_test)
                test_EB_wellspec_thetahat = Z_test + sigma_test**2 * test_EB_wellspec_score
                test_MSE_wellspec_list.append(np.linalg.norm(test_EB_wellspec_thetahat.cpu().detach().numpy() - theta_test)**2 / n)
                test_SURE_wellspec_list.append(model_wellspec.opt_func_SURE(Z_test, test_Ft_Rep, X_test).cpu().detach().numpy())

                print("Finished evaluation on test data")

    mse_sure_df = pd.DataFrame({'n': 2*ns_list,
                                'hidden_sizes': 2*hidden_sizes_simulation_list,
                              'experiment': 2*experiments_list,
                              'MSE_wellspec': train_MSE_wellspec_list + test_MSE_wellspec_list,
                              'SURE_wellspec': train_SURE_wellspec_list + test_SURE_wellspec_list,
                              'data': len(ns)*len(experiments)*len(hidden_sizes_list)*['train'] + len(ns)*len(experiments)*len(hidden_sizes_list)*['test']}) 
    
        
    return(mse_sure_df)

def simulate_location_scale(ns=[100, 200, 400, 800, 1600, 3200, 6400],
                          experiments=["c", "d", "d5", "e", "f"],
                          hidden_sizes=(8,8), B=100,
                          optimizer_str="adam",
                          simulate_location_list=[True],
                          simulate_scale_list=[True], 
                          skip_connect=False):
    

    experiments_list = []
    ns_list = []
    use_locations_list =[]
    use_scales_list = []

    train_MSE_wellspec_list = [] 
    train_SURE_wellspec_list = []

    train_MSE_misspec_list = []
    train_SURE_misspec_list = []

    train_MSE_NPMLE_list = [] 
    train_SURE_NPMLE_list = [] 

    train_MSE_NPMLEinit_list = [] 
    train_SURE_NPMLEinit_list = [] 

    train_MSE_thetaG_list = []
    train_SURE_thetaG_list = []

    train_MSE_surels_list = []
    train_SURE_surels_list = []

    train_MSE_ebcf_list = []
    test_MSE_ebcf_list = [] 
    
    test_MSE_wellspec_list = [] 
    test_SURE_wellspec_list = []

    test_MSE_misspec_list = []
    test_SURE_misspec_list = []

    test_MSE_NPMLE_list = [] 
    test_SURE_NPMLE_list = [] 

    test_MSE_NPMLEinit_list = [] 
    test_SURE_NPMLEinit_list = [] 

    test_MSE_thetaG_list = []
    test_SURE_thetaG_list = []

    test_MSE_surels_list = []
    test_SURE_surels_list = []

    for n in ns:

        print(f"n: {n}")

        for experiment in experiments:

            print(f"experiment: {experiment}")

            theta, Z, X = simulate_data.xie(experiment=experiment, n=n)

            theta_test, Z_test, X_test = simulate_data.xie(experiment=experiment, n=n)
            sigma_test = X_test[:,-1]

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
                # model_NPMLE is CPU
                print(f"Finished solving NPMLE, with SURE {SURE_NPMLE}")
                print(f"Finished solving NPMLE, with in-sample MSE {twonorm_diff_NPMLE / n}")
                
                found_NPMLE_solution = True

                # NPMLE evaluation
                Z_test_cpu = Z_test.cpu()
                sigma_test_cpu = sigma_test.cpu()
                test_NPMLE_score = model_NPMLE.compute_score(Z_test_cpu, n, B, sigma_test_cpu) 
                test_NPMLE_thetahat = Z_test_cpu + sigma_test_cpu**2 * test_NPMLE_score

            except Exception as e:
                print(f"Mosek failed on this run") 

            # Xie - theta G parametric baseline
            theta_hat_G, MSE_thetaG, SURE_thetaG, grand_mean, lambda_hat = models.theta_hat_G(theta, Z, X)
            print(f"Finished solving parametric estimator, with SURE {SURE_thetaG}")
            print(f"Finished solving parametric estimator, with in-sample MSE {MSE_thetaG}")
            
            # parametric evaluation
            test_theta_hat_G, test_MSE_thetaG, test_SURE_thetaG, grand_mean, lambda_hat = models.theta_hat_G(theta_test, Z_test, X_test)

            # EBCF 
            theta_hats_ebcf, A_hats_ebcf, MSE_ebcf, model_ebcf = train.train_EBCF(X, Z, theta) 
            print(f"Finished solving EBCF, with in-sample MSE {MSE_ebcf}") 

            # EBCF evaluation
            Z_test_hat = model_ebcf.get_Z_hat(Z_test, X_test) 
            test_theta_hat_EBCF, test_MSE_EBCF, test_SURE_EBCF, test_A_hat_EBCF = models.theta_hat_EBCF(theta_test, Z_test, X_test, Z_test_hat) 

            # EB - SURE LS
            result_surels = train.train_sure_ls(X, Z, theta, set_seed = None, d=2, device=simulate_data.device, 
                                                optimizer_str=optimizer_str, hidden_sizes=hidden_sizes) 
            model_surels, Ft_Rep_surels, SURE_surels, score_surels, theta_hats_surels, twonorm_diff_surels = result_surels
            SURE_surels = SURE_surels[-1] 
            print(f"Finished training EB SURE-LS, with SURE: {SURE_surels}")
            print(f"Finished training EB SURE-LS, with in-sample MSE: {twonorm_diff_surels / n}") 

            # SURE LS evaluation 
            test_Ft_Rep = model_surels.feature_representation(X_test) 
            test_EB_surels_score = model_surels.compute_score(Z_test, test_Ft_Rep, X_test)
            test_EB_surels_thetahat = Z_test + sigma_test**2 * test_EB_surels_score 

            for use_scale in simulate_scale_list:

                print(f"use_scale: {use_scale}")

                for use_location in simulate_location_list:

                    print(f"use_location: {use_location}")

                    ns_list.append(n)
                    experiments_list.append(experiment)
                    use_locations_list.append(use_location)
                    use_scales_list.append(use_scale)

                    if found_NPMLE_solution:
                        # EB - NPMLEinit
                        result_NPMLEinit = train.train_no_covariates(n, B, Z, theta, X, opt_objective = 'both', init_val_pi = tr.log(pi_hat_NPMLE),
                                                                    optimizer_str=optimizer_str, use_location=use_location, use_scale=use_scale, device=simulate_data.device) 
                        model_NPMLEinit, SURE_NPMLEinit, score_NPMLEinit, theta_hats_NPMLEinit, twonorm_diff_NPMLEinit = result_NPMLEinit
                        SURE_NPMLEinit = SURE_NPMLEinit[-1]
                        print(f"Finished training EB misspecified with NPMLE init, with SURE: {SURE_NPMLEinit}")
                        print(f"Finished training EB misspecified with NPMLE init, with in-sample MSE: {twonorm_diff_NPMLEinit / n}")

                    # EB - misspecified
                    result_misspec = train.train_no_covariates(n, B, Z, theta, X, opt_objective = 'both',
                                                            optimizer_str=optimizer_str, use_scale=use_scale, use_location=use_location, device=simulate_data.device) 
                    model_misspec, SURE_misspec, score_misspec, theta_hats_misspec, twonorm_diff_misspec = result_misspec
                    SURE_misspec = SURE_misspec[-1]
                    print(f"Finished training EB misspecified, with SURE: {SURE_misspec}")
                    print(f"Finished training EB misspecified, with in-sample MSE: {twonorm_diff_misspec / n}")

                    # EB - well specified
                    result_wellspec = train.train_covariates(X, Z, theta, objective="SURE", set_seed=None, d=2, B=B, drop_sigma=False,
                                                                optimizer_str=optimizer_str,
                                                                hidden_sizes=hidden_sizes, use_location=use_location, use_scale=use_scale, device=simulate_data.device, 
                                                                skip_connect=skip_connect) 
                    model_wellspec, Ft_Rep_wellspec, SURE_wellspec, NLL_wellspec, score_wellspec, theta_hats_wellspec, twonorm_diff_wellspec = result_wellspec
                    SURE_wellspec = SURE_wellspec[-1]
                    NLL_wellspec = NLL_wellspec[-1]
                    print(f"Finished training EB wellspecified, with SURE: {SURE_wellspec}")
                    print(f"Finished training EB wellspecified, with in-sample MSE: {twonorm_diff_wellspec / n}")

                    # Append train results
                    train_MSE_misspec_list.append(twonorm_diff_misspec/n)
                    train_SURE_misspec_list.append(SURE_misspec)

                    train_MSE_wellspec_list.append(twonorm_diff_wellspec/n)
                    train_SURE_wellspec_list.append(SURE_wellspec)

                    train_MSE_thetaG_list.append(MSE_thetaG)
                    train_SURE_thetaG_list.append(SURE_thetaG)

                    train_MSE_surels_list.append(twonorm_diff_surels/n)
                    train_SURE_surels_list.append(SURE_surels) 

                    train_MSE_ebcf_list.append(MSE_ebcf) 

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


                    # Test results

                    # EB misspec evaluation
                    test_EB_misspec_score = model_misspec.compute_score(Z_test, n, B, sigma_test)
                    test_EB_misspec_thetahat = Z_test + sigma_test**2 * test_EB_misspec_score
                    test_MSE_misspec_list.append(np.linalg.norm(test_EB_misspec_thetahat.cpu().detach().numpy() - theta_test)**2 / n)
                    test_SURE_misspec_list.append(model_misspec.opt_func(Z_test, n, B, sigma_test).cpu().detach().numpy())

                    # EB wellspec evaluation
                    test_Ft_Rep = model_wellspec.feature_representation(X_test)
                    test_EB_wellspec_score = model_wellspec.compute_score(Z_test, test_Ft_Rep, X_test)
                    test_EB_wellspec_thetahat = Z_test + sigma_test**2 * test_EB_wellspec_score
                    test_MSE_wellspec_list.append(np.linalg.norm(test_EB_wellspec_thetahat.cpu().detach().numpy() - theta_test)**2 / n)
                    test_SURE_wellspec_list.append(model_wellspec.opt_func_SURE(Z_test, test_Ft_Rep, X_test).cpu().detach().numpy())

                    # parametric evaluation
                    test_MSE_thetaG_list.append(test_MSE_thetaG)
                    test_SURE_thetaG_list.append(test_SURE_thetaG)

                    # EB SURE-LS evaluation
                    test_MSE_surels_list.append(np.linalg.norm(test_EB_surels_thetahat.cpu().detach().numpy() - theta_test)**2 / n)
                    test_SURE_surels_list.append(model_surels.opt_func_SURE(Z_test, test_Ft_Rep, X_test).cpu().detach().numpy()) 

                    # EBCF evaluation
                    test_MSE_ebcf_list.append(test_MSE_EBCF) 

                    if found_NPMLE_solution:
                        test_MSE_NPMLE_list.append(np.linalg.norm(test_NPMLE_thetahat.detach().numpy() - theta_test)**2 / n)
                        test_SURE_NPMLE_list.append(model_NPMLE.opt_func(Z_test_cpu, n, B, sigma_test_cpu).detach().numpy())

                        # NPMLEinit evaluation
                        test_EB_NPMLEinit_score = model_NPMLEinit.compute_score(Z_test, n, B, sigma_test)
                        test_EB_NPMLEinit_thetahat = Z_test + sigma_test**2 * test_EB_NPMLEinit_score
                        test_MSE_NPMLEinit_list.append(np.linalg.norm(test_EB_NPMLEinit_thetahat.cpu().detach().numpy() - theta_test)**2 / n)
                        test_SURE_NPMLEinit_list.append(model_NPMLEinit.opt_func(Z_test, n, B, sigma_test).cpu().detach().numpy())
                    else:
                        test_MSE_NPMLE_list.append(None)
                        test_MSE_NPMLEinit_list.append(None)

                        test_SURE_NPMLE_list.append(None)
                        test_SURE_NPMLEinit_list.append(None)

            print("Finished evaluation on test data")

    mse_sure_df = pd.DataFrame({'n': 2*ns_list,
                                'use_location': 2*use_locations_list,
                                'use_scale': 2*use_scales_list,
                                'experiment': 2*experiments_list,
                                'MSE_wellspec': train_MSE_wellspec_list + test_MSE_wellspec_list,
                                'MSE_misspec': train_MSE_misspec_list + test_MSE_misspec_list,
                                'MSE_surels': train_MSE_surels_list + train_MSE_surels_list,
                                'MSE_NPMLEinit': train_MSE_NPMLEinit_list + test_MSE_NPMLEinit_list,
                                'MSE_NPMLE': train_MSE_NPMLE_list + test_MSE_NPMLE_list,
                                'MSE_thetaG': train_MSE_thetaG_list + test_MSE_thetaG_list,
                                'MSE_EBCF': train_MSE_ebcf_list + test_MSE_ebcf_list, 
                                'SURE_wellspec': train_SURE_wellspec_list + test_SURE_wellspec_list,
                                'SURE_misspec': train_SURE_misspec_list + test_SURE_misspec_list, 
                                'SURE_surels': train_SURE_surels_list + test_SURE_surels_list,
                                'SURE_NPMLEinit': train_SURE_NPMLEinit_list + test_SURE_NPMLEinit_list,
                                'SURE_NPMLE': train_SURE_NPMLE_list + test_SURE_NPMLE_list,
                                'SURE_thetaG': train_SURE_thetaG_list + test_SURE_thetaG_list,
                                'data': len(ns)*len(experiments)*len(simulate_location_list)*len(simulate_scale_list)*['train'] + len(ns)*len(experiments)*len(simulate_location_list)*len(simulate_scale_list)*['test']}) 
    
    return(mse_sure_df)
    




def make_df(ns=[100, 200, 400, 800, 1600, 3200, 6400], 
            experiments = ["c", "d", "d5", "e", "f"], 
            optimizer_str="adam", m_sim=300, B=100,
            hidden_sizes=(8,8), hidden_sizes_list=None,
            compare_hidden_sizes=False,
            compare_location_scale=False,
            simulate_location_list=[True],
            simulate_scale_list=[True], 
            skip_connect=False):
    """
    Compute MSEs and SURE on train and test for all models, m_sim times
    """

    mse_sure_results = [] # list of dataframes
    
    if hidden_sizes_list == None:
        for m in range(m_sim):
            print(f"m_sim: {m}")

            if compare_hidden_sizes:
                mse_sure_results.append(simulate_nn_sizes(ns=ns, experiments=experiments,
                                                        optimizer_str=optimizer_str))
            elif compare_location_scale:
                mse_sure_results.append(simulate_location_scale(ns=ns, experiments=experiments,
                                                                hidden_sizes=hidden_sizes, B=B,
                                                            optimizer_str=optimizer_str,
                                                            simulate_location_list=simulate_location_list,
                                                            simulate_scale_list=simulate_scale_list, 
                                                            skip_connect=skip_connect)) 
            else:
                mse_sure_results.append(get_simulation_result(ns=ns, experiments=experiments,
                                                    optimizer_str=optimizer_str)) 
    else: 
        for m in range(m_sim):
            print(f"m_sim: {m}")

            # for hidden_sizes in hidden_sizes_list: 
            #     print(f"hidden_size: {hidden_sizes}")

            #     df_results = simulate_location_scale(ns=ns, experiments=experiments,
            #                                          hidden_sizes=hidden_sizes, B=B,
            #                                          optimizer_str=optimizer_str,
            #                                          simulate_location_list=simulate_location_list,
            #                                          simulate_scale_list=simulate_scale_list)
            #     df_results['hidden_sizes'] = (df_results.shape[0])*[hidden_sizes] 
            #     mse_sure_results.append(df_results)
            mse_sure_results.append(simulate_nn_sizes(ns=ns, experiments=experiments,
                                                        optimizer_str=optimizer_str, 
                                                        hidden_sizes_list=hidden_sizes_list, 
                                                        skip_connect=skip_connect)) 

    mse_sure_df = pd.concat(mse_sure_results)

    return mse_sure_df


def train_and_save_models(n=10000, B=100, experiments = ["c", "d", "d5", "e", "f"], optimizer_str="adam", activation_fn_str="relu",
                verbose=False):
    """
    optimizer and activation function comparisons

    * optimizer_str: describes the optimizer used. for optimizers that AREN'T "adam",
                     save the model output in a <experiment>_<optimizer_str>
    """

    for experiment in experiments:

        print(f"experiment: {experiment}")
        print(optimizer_str)
        print(activation_fn_str)

        if activation_fn_str == "relu": 
            activation_fn = tr.nn.ReLU()
        elif activation_fn_str == "elu": 
            activation_fn = tr.nn.ELU()
        elif activation_fn_str == "silu": 
            activation_fn = tr.nn.SiLU() 

        experiment_str = experiment + "_" + optimizer_str + "_" + activation_fn_str + "/"
        print(experiment_str)

        count = 0
        count_exceptions = 0

        rn.seed(600)
        random.seed(50)
        tr.manual_seed(100)

        while count < 1:
            try: 
                theta, Z, X = simulate_data.xie(experiment=experiment, n=n)
                np.save("models/xie/" + experiment_str +  "/theta", theta)
                tr.save(Z, "models/xie/" + experiment_str +  "/Z")
                tr.save(X, "models/xie/" + experiment_str +  "/X")
                # print("from simulate data")
                # print(f"Z_theta.shape: {Z_theta.shape}")
                # print(f"Z_theta_by_sigma_sq.shape: {Z_theta_by_sigma_sq.shape}")

                # NPMLE - misspecified
                result_NPMLE = train.train_npmle(n, B, Z, theta, X) 
                problem_NPMLE, loss_NPMLE, score_NPMLE, theta_hat_NPMLE, twonorm_diff_NPMLE, pi_hat_NPMLE = result_NPMLE
                pi_hat_NPMLE[pi_hat_NPMLE < 0] = 0
                np.save("models/xie/" + experiment_str +  "/pi_hat_NPMLE", pi_hat_NPMLE.detach().numpy())

                count += 1

            except Exception as e:
                count_exceptions +=  1
                print(f"Error occurred") 

                if count_exceptions == 8:
                    count += 1
                
        # EB - misspecified (with NPMLE optmized pi as initial) 
        result_NPMLEinit = train.train_no_covariates(n, B, Z, theta, X, opt_objective = 'both', init_val_pi = tr.log(pi_hat_NPMLE),
                                                        optimizer_str=optimizer_str) 
        model_NPMLEinit, SURE_NPMLEinit, score_NPMLEinit, theta_hats_NPMLEinit, twonorm_diff_NPMLEinit = result_NPMLEinit
        # SURE_NPMLEinit = SURE_NPMLEinit[-1]
        tr.save(model_NPMLEinit.state_dict(), "models/xie/" + experiment_str +  "/model_EB_NPMLEinit")
        tr.save(SURE_NPMLEinit, "models/xie/" + experiment_str +  "/SURE_NPMLEinit")

        # EB - well specified
        result_wellspec = train.train_covariates(X, Z, theta, objective="SURE", set_seed=None, d=2, B=100, drop_sigma=False,
                                                                optimizer_str=optimizer_str, activation_fn = activation_fn)
        model_wellspec, Ft_Rep_wellspec, SURE_wellspec, NLL_wellspec, score_wellspec, theta_hats_wellspec, twonorm_diff_wellspec = result_wellspec
        if verbose:
            print(f"Optimized SURE_misspec: {SURE_wellspec[-1]}")

        NLL_wellspec = NLL_wellspec[-1]
        tr.save(model_wellspec.state_dict(), "models/xie/" + experiment_str +  "/model_EB_wellspec")
        # for param_tensor in model_wellspec.state_dict():
        #     print(param_tensor, "\t", model_wellspec.state_dict()[param_tensor].size())
        tr.save(Ft_Rep_wellspec, "models/xie/" + experiment_str +  "/Ft_Rep_wellspec")
        tr.save(SURE_wellspec, "models/xie/" + experiment_str +  "/SURE_wellspec")

        # EB - misspecified
        result_misspec = train.train_no_covariates(n, B, Z, theta, X, opt_objective = 'both',
                                                                optimizer_str=optimizer_str) 
        model_misspec, SURE_misspec, score_misspec, theta_hats_misspec, twonorm_diff_misspec = result_misspec
        
        if verbose:
            print(f"Optimized SURE_misspec: {SURE_misspec[-1]}")

        # SURE_misspec = SURE_misspec[-1]
        tr.save(model_misspec.state_dict(), "models/xie/" + experiment_str +  "/model_EB_misspec")
        tr.save(SURE_misspec, "models/xie/" + experiment_str +  "/SURE_misspec")

        # SURE - ground truth
        SURE_truth = tr.tensor(0)
        if experiment == 'e': 
            varA = X**2
            print(n); print(varA.min()); print(varA.max())  
            SURE_truth = tr.sum(varA)/(2*n) 
            print(SURE_truth) 
            tr.save(SURE_truth, "models/xie/" + experiment_str +  "/SURE_truth") 

        # Xie - theta G parametric baseline
        theta_hat_G, MSE_thetaG, SURE_thetaG, grand_mean, lambda_hat = models.theta_hat_G(theta, Z, X)
        np.save("models/xie/" + experiment_str +  "/theta_hat_G", theta_hat_G)
        np.save("models/xie/" + experiment_str +  "/grand_mean", grand_mean)
        np.save("models/xie/" + experiment_str +  "/lambda_hat", lambda_hat)

def train_and_save_models_location_scale(seeds,
                                         n=6400, 
                                         hidden_sizes=(8,8),
                                         B=100,
                                         experiments = ["c", "d", "d5", "e", "f", "g"],
                                         simulate_location_list=[True, False],
                                         simulate_scale_list=[True, False],
                                         optimizer_str="adam", path="models/xie_location_scale"):
    """
    
    """

    path = path + "_" + ''.join(str(seed) for seed in seeds) + "/"

    for experiment in experiments:

        print(f"experiment: {experiment}")
        
        experiment_str = experiment 
        
        print(experiment_str)

        rn.seed(seeds[0])
        random.seed(seeds[1])
        tr.manual_seed(seeds[2])
        # 600 50 100
        # 374 55 999 

        # Simulate and save data
        theta, Z, X = simulate_data.xie(experiment=experiment, n=n)
        np.save(path + experiment_str +  "/theta", theta)
        tr.save(Z, path + experiment_str +  "/Z")
        tr.save(X, path + experiment_str +  "/X")

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
            # model_NPMLE is CPU
            print(f"Finished solving NPMLE, with SURE {SURE_NPMLE}")
            print(f"Finished solving NPMLE, with in-sample MSE {twonorm_diff_NPMLE / n}")
            np.save(path + experiment_str +  "/pi_hat_NPMLE", pi_hat_NPMLE.detach().numpy())
            
            found_NPMLE_solution = True

        except Exception as e:
            print(f"Mosek failed on this run") 

        # Xie - theta G parametric baseline
        theta_hat_G, MSE_thetaG, SURE_thetaG, grand_mean, lambda_hat = models.theta_hat_G(theta, Z, X)
        print(f"Finished solving parametric estimator, with SURE {SURE_thetaG}")
        print(f"Finished solving parametric estimator, with in-sample MSE {MSE_thetaG}")
        np.save(path + experiment_str +  "/theta_hat_G", theta_hat_G)
        np.save(path + experiment_str +  "/grand_mean", grand_mean)
        np.save(path + experiment_str +  "/lambda_hat", lambda_hat)
        

        for use_scale in simulate_scale_list:

            print(f"use_scale: {use_scale}")

            for use_location in simulate_location_list:

                print(f"use_location: {use_location}")

                if use_location & use_scale:
                    suffix = "_location_scale"
                elif use_location & (not use_scale): 
                    suffix = "_location_IQR"
                elif (not use_location) & use_scale:
                    suffix = "_median_scale"
                else:
                    suffix = "_median_IQR"

                if found_NPMLE_solution:
                    # EB - NPMLEinit
                    result_NPMLEinit = train.train_no_covariates(n, B, Z, theta, X, opt_objective = 'both', init_val_pi = tr.log(pi_hat_NPMLE),
                                                                optimizer_str=optimizer_str, use_location=use_location, use_scale=use_scale, device=simulate_data.device) 
                    model_NPMLEinit, SURE_NPMLEinit, score_NPMLEinit, theta_hats_NPMLEinit, twonorm_diff_NPMLEinit = result_NPMLEinit
                    print(f"Finished training EB misspecified with NPMLE init, with SURE: {SURE_NPMLEinit[-1]}")
                    print(f"Finished training EB misspecified with NPMLE init, with in-sample MSE: {twonorm_diff_NPMLEinit / n}")
                    tr.save(model_NPMLEinit.state_dict(), path + experiment_str +  "/model_EB_NPMLEinit" + suffix)
                    tr.save(SURE_NPMLEinit, path + experiment_str +  "/SURE_NPMLEinit"+ suffix)

                # EB - misspecified
                result_misspec = train.train_no_covariates(n, B, Z, theta, X, opt_objective = 'both',
                                                        optimizer_str=optimizer_str, use_scale=use_scale, use_location=use_location, device=simulate_data.device) 
                model_misspec, SURE_misspec, score_misspec, theta_hats_misspec, twonorm_diff_misspec = result_misspec
                print(f"Finished training EB misspecified, with SURE: {SURE_misspec[-1]}")
                print(f"Finished training EB misspecified, with in-sample MSE: {twonorm_diff_misspec / n}")
                tr.save(model_misspec.state_dict(), path + experiment_str +  "/model_EB_misspec" + suffix)
                tr.save(SURE_misspec, path + experiment_str +  "/SURE_misspec" + suffix)
                

                # EB - well specified
                result_wellspec = train.train_covariates(X, Z, theta, objective="SURE", set_seed=None, d=2, B=B, drop_sigma=False,
                                                            optimizer_str=optimizer_str,
                                                            hidden_sizes=hidden_sizes, use_location=use_location, use_scale=use_scale, device=simulate_data.device)
                model_wellspec, Ft_Rep_wellspec, SURE_wellspec, NLL_wellspec, score_wellspec, theta_hats_wellspec, twonorm_diff_wellspec = result_wellspec
                print(f"Finished training EB wellspecified, with SURE: {SURE_wellspec[-1]}")
                print(f"Finished training EB wellspecified, with in-sample MSE: {twonorm_diff_wellspec / n}")
                tr.save(model_wellspec.state_dict(), path + experiment_str +  "/model_EB_wellspec" + suffix)
                tr.save(Ft_Rep_wellspec, path + experiment_str +  "/Ft_Rep_wellspec" + suffix)
                tr.save(SURE_wellspec, path + experiment_str +  "/SURE_wellspec"+ suffix)

        # EB - surels
        result_surels = train.train_sure_ls(X, Z, theta, set_seed = None, d=2, device=simulate_data.device, 
                                            optimizer_str=optimizer_str, hidden_sizes=hidden_sizes) 
        model_surels, Ft_Rep_surels, SURE_surels, score_surels, theta_hats_surels, twonorm_diff_surels = result_surels
        print(f"Finished training EB SURE-LS, with SURE: {SURE_surels[-1]}")
        print(f"Finished training EB SURE-LS, with in-sample MSE: {twonorm_diff_surels / n}") 
        tr.save(model_surels.state_dict(), path + experiment_str +  "/model_EB_surels")
        tr.save(Ft_Rep_surels, path + experiment_str +  "/Ft_Rep_surels")
        tr.save(SURE_surels, path + experiment_str +  "/SURE_surels")
    
        # SURE - ground truth
        SURE_truth = tr.tensor(0)
        if experiment == 'e': 
            varA = X**2
            print(n); print(varA.min()); print(varA.max())  
            SURE_truth = tr.sum(((Z - 2.5 + 5*varA)/2)**2 - (varA)) / n 
        tr.save(SURE_truth, path + experiment_str +  "/SURE_truth") 

def train_and_save_models_failures(n=10000, B=100, experiments = ["c", "d", "d5", "e", "f"], optimizer_str="adam", activation_fn = tr.nn.ReLU(), 
                         verbose=False, model_fails="misspec"):
    """
    * optimizer_str: describes the optimizer used. for optimizers that AREN'T "adam",
                     save the model output in a <experiment>_<optimizer_str>
    * model_fails: "misspec", "wellspec" 
    """

    for experiment in experiments:

            print(f"experiment: {experiment}")
            
            if optimizer_str != "adam":
                experiment_str = experiment + "_" + optimizer_str + "/"
            else:
                experiment_str = experiment

            

            is_SURE_NA = False
            while_count = 0

            while (not is_SURE_NA):

                print(f"Searching for SURE NA: {while_count}th time")

                set_seed = rn.randint(low=1, high=2**32-1)

                count = 0
                count_exceptions = 0

                while count < 1:
                    try: 

                        theta, Z, X = simulate_data.xie(experiment=experiment, n=n)
                        # print("from simulate data")
                        # print(f"Z_theta.shape: {Z_theta.shape}")
                        # print(f"Z_theta_by_sigma_sq.shape: {Z_theta_by_sigma_sq.shape}")

                        # NPMLE - misspecified
                        result_NPMLE = train.train_npmle(n, B, Z, theta, X) 
                        problem_NPMLE, loss_NPMLE, score_NPMLE, theta_hat_NPMLE, twonorm_diff_NPMLE, pi_hat_NPMLE = result_NPMLE
                        pi_hat_NPMLE[pi_hat_NPMLE < 0] = 0
                        print("Finished solving NPMLE")

                        # EB - misspecified (with NPMLE optmized pi as initial) 
                        result_NPMLEinit = train.train_no_covariates(n, B, Z, theta, X, opt_objective = 'both', init_val_pi = tr.log(pi_hat_NPMLE),
                                                                    optimizer_str=optimizer_str) 
                        model_NPMLEinit, SURE_NPMLEinit, score_NPMLEinit, theta_hats_NPMLEinit, twonorm_diff_NPMLEinit = result_NPMLEinit
                        # SURE_NPMLEinit = SURE_NPMLEinit[-1]
                        print("Finished training EB misspecified with NPMLE init")

                        count += 1

                    except Exception as e:
                        count_exceptions +=  1
                        print(f"Error occurred") 

                        if count_exceptions == 8:
                            count += 1
                        
                        
                # EB - well specified
                result_wellspec = train.train_covariates(X, Z, theta, objective="SURE", set_seed=set_seed, d=2, B=100, drop_sigma=False,
                                                            optimizer_str=optimizer_str, activation_fn=activation_fn)
                model_wellspec, Ft_Rep_wellspec, SURE_wellspec, NLL_wellspec, score_wellspec, theta_hats_wellspec, twonorm_diff_wellspec = result_wellspec
                print("Finished training EB well specified")
                if verbose:
                    print(f"Optimized SURE_wellspec: {SURE_wellspec[-1]}")

                # EB - misspecified
                result_misspec = train.train_no_covariates(n, B, Z, theta, X, opt_objective = 'both',
                                                                    optimizer_str=optimizer_str) 
                model_misspec, SURE_misspec, score_misspec, theta_hats_misspec, twonorm_diff_misspec = result_misspec
                print("Finished training EB misspecified")
                if verbose:
                    print(f"Optimized SURE_misspec: {SURE_misspec[-1]}")


                # Xie - theta G parametric baseline
                theta_hat_G, MSE_thetaG, SURE_thetaG, grand_mean, lambda_hat = models.theta_hat_G(theta, Z, X)
                print("Finished solving parametric")

                if model_fails == "misspec":
                    is_SURE_NA = np.isnan(SURE_misspec[-1])
                elif model_fails == "wellspec":
                    is_SURE_NA = np.isnan(SURE_wellspec[-1])
                else:
                    print("typo in <model_fails>")
                    is_SURE_NA = True
                while_count += 1
            
            # Save objects after finding NA
            
            np.save("models/xie_failures/" + model_fails + "/" + experiment_str +  "/theta", theta)
            tr.save(Z, "models/xie_failures/" + model_fails + "/" + experiment_str +  "/Z")
            tr.save(X, "models/xie_failures/" + model_fails + "/" + experiment_str +  "/X")
            
            np.save("models/xie_failures/" + model_fails + "/" + experiment_str +  "/pi_hat_NPMLE", pi_hat_NPMLE.detach().numpy())

            tr.save(model_NPMLEinit.state_dict(), "models/xie_failures/" + model_fails + "/" + experiment_str +  "/model_EB_NPMLEinit")
            tr.save(SURE_NPMLEinit, "models/xie_failures/" + model_fails + "/" + experiment_str +  "/SURE_NPMLEinit")

            np.save("models/xie_failures/" + model_fails + "/" + experiment_str + "/set_seed", set_seed)
            tr.save(model_wellspec.state_dict(), "models/xie_failures/" + model_fails + "/" + experiment_str +  "/model_EB_wellspec")
            tr.save(Ft_Rep_wellspec, "models/xie_failures/" + model_fails + "/" + experiment_str +  "/Ft_Rep_wellspec")
            tr.save(SURE_wellspec, "models/xie_failures/" + model_fails + "/" + experiment_str +  "/SURE_wellspec")
        
            tr.save(model_misspec.state_dict(), "models/xie_failures/" + model_fails + "/" + experiment_str +  "/model_EB_misspec")
            tr.save(SURE_misspec, "models/xie_failures/" + model_fails + "/" + experiment_str +  "/SURE_misspec")

            np.save("models/xie_failures/" + model_fails + "/" + experiment_str +  "/theta_hat_G", theta_hat_G)
            np.save("models/xie_failures/" + model_fails + "/" + experiment_str +  "/grand_mean", grand_mean)
            np.save("models/xie_failures/" + model_fails + "/" + experiment_str +  "/lambda_hat", lambda_hat)




def bfgs_versus_adam(n=10000, B=100, experiments = ["d", "d5"], model="misspec", m_sim=30):
    """
    When model="misspec", compare Adam versus BFGS on misspecified case
    When model="wellspec" compare Adam versus BFGS on wellspecified case 

    Save train and test MSEs, SUREs.

    When an NA is encountered, save the seed, dataset, and model.
    """

    # Initialize lists for results
    experiments_list = []

    if model == "misspec":
        train_MSE_adam_misspec_list = [] # EB no covariates
        train_MSE_adam_NPMLEinit_list = [] # EB no covariates with NPMLE initialization
        test_MSE_adam_misspec_list = []
        test_MSE_adam_NPMLEinit_list = []

        train_SURE_adam_misspec_list = []
        train_SURE_adam_NPMLEinit_list = []
        test_SURE_adam_misspec_list = []
        test_SURE_adam_NPMLEinit_list = []

        train_MSE_bfgs_misspec_list = [] 
        train_MSE_bfgs_NPMLEinit_list = [] 
        test_MSE_bfgs_misspec_list = []
        test_MSE_bfgs_NPMLEinit_list = []

        train_SURE_bfgs_misspec_list = []
        train_SURE_bfgs_NPMLEinit_list = []
        test_SURE_bfgs_misspec_list = []
        test_SURE_bfgs_NPMLEinit_list = []

    elif model == "wellspec":

        found_NA_ELU = False 
        found_NA_SiLU = False

        train_adam_MSE_wellspec_list = [] # well specified EB
        test_adam_MSE_wellspec_list = [] 
        train_adam_SURE_wellspec_list = []
        test_adam_SURE_wellspec_list = []

        train_ELU_MSE_wellspec_list = [] 
        test_ELU_MSE_wellspec_list = []
        train_ELU_SURE_wellspec_list = []
        test_ELU_SURE_wellspec_list = []

        train_SiLU_MSE_wellspec_list = [] 
        test_SiLU_MSE_wellspec_list = []
        train_SiLU_SURE_wellspec_list = []
        test_SiLU_SURE_wellspec_list = []

    else:
        print("typo in model")

    for sim in range(m_sim):
        print(f"sim: {sim}")

        for experiment in experiments:
            print(f"experiment: {experiment}")
            experiments_list.append(experiment)

            if model=="misspec":
        
                # count MOSEK failures
                count = 0
                count_exceptions = 0

                # EB - NPMLEinit
                while count < 1:
                    try: 
                        # Train data
                        theta, Z, X = simulate_data.xie(experiment=experiment, n=n)
                        # print("from simulate data")
                        # print(f"Z_theta.shape: {Z_theta.shape}")
                        # print(f"Z_theta_by_sigma_sq.shape: {Z_theta_by_sigma_sq.shape}")

                        # NPMLE - misspecified. 
                        result_NPMLE = train.train_npmle(n, B, Z, theta, X) 
                        problem_NPMLE, loss_NPMLE, score_NPMLE, theta_hat_NPMLE, twonorm_diff_NPMLE, pi_hat_NPMLE = result_NPMLE
                        pi_hat_NPMLE[pi_hat_NPMLE < 0] = 0
                        # model_NPMLE = models.model_pi_sure(Z=Z, B=B, init_val=tr.log(pi_hat_NPMLE), device="cpu") # to compute SURE, theta hat
                        # model_NPMLE is CPU
                        print("Finished solving NPMLE")

                        # EB - misspecified (with NPMLE optmized pi as initial) 
                        ## Adam
                        adam_result_NPMLEinit = train.train_no_covariates(n, B, Z, theta, X, opt_objective = 'both', init_val_pi = tr.log(pi_hat_NPMLE),
                                                                    optimizer_str="adam") 
                        adam_model_NPMLEinit, adam_SURE_NPMLEinit, adam_score_NPMLEinit, adam_theta_hats_NPMLEinit, adam_twonorm_diff_NPMLEinit = adam_result_NPMLEinit
                        adam_SURE_NPMLEinit = adam_SURE_NPMLEinit[-1]
                        print("Finished training EB misspecified with NPMLE init, Adam")

                        ## bfgs
                        bfgs_result_NPMLEinit = train.train_no_covariates(n, B, Z, theta, X, opt_objective = 'both', init_val_pi = tr.log(pi_hat_NPMLE),
                                                                    optimizer_str="bfgs") 
                        bfgs_model_NPMLEinit, bfgs_SURE_NPMLEinit, bfgs_score_NPMLEinit, bfgs_theta_hats_NPMLEinit, bfgs_twonorm_diff_NPMLEinit = bfgs_result_NPMLEinit
                        bfgs_SURE_NPMLEinit = bfgs_SURE_NPMLEinit[-1]
                        print("Finished training EB misspecified with NPMLE init, bfgs")
                        count += 1

                    except Exception as e:
                        count_exceptions +=  1
                        print(f"Error occurred") 

                        if count_exceptions == 5:
                            count += 1
                    
                # EB - misspecified
                # Train Adam
                adam_result_misspec = train.train_no_covariates(n, B, Z, theta, X, opt_objective = 'both',
                                                                optimizer_str="bfgs") 
                adam_model_misspec, adam_SURE_misspec, adam_score_misspec, adam_theta_hats_misspec, adam_twonorm_diff_misspec = adam_result_misspec
                print("Finished training EB misspecified, adam")

                # Train bfgs
                bfgs_result_misspec = train.train_no_covariates(n, B, Z, theta, X, opt_objective = 'both',
                                                                optimizer_str="bfgs") 
                bfgs_model_misspec, bfgs_SURE_misspec, bfgs_score_misspec, bfgs_theta_hats_misspec, bfgs_twonorm_diff_misspec = bfgs_result_misspec
                print("Finished training EB misspecified, bfgs")

                # Save training results
                if count_exceptions < 5:
                    # NPMLE init
                    train_MSE_adam_NPMLEinit_list.append(adam_twonorm_diff_NPMLEinit/n) 
                    train_SURE_adam_NPMLEinit_list.append(adam_SURE_NPMLEinit) 

                    train_MSE_bfgs_NPMLEinit_list.append(bfgs_twonorm_diff_NPMLEinit/n) 
                    train_SURE_bfgs_NPMLEinit_list.append(bfgs_SURE_NPMLEinit) 
                else:
                    train_MSE_adam_NPMLEinit_list.append(-100) 
                    train_SURE_adam_NPMLEinit_list.append(-100) 

                    train_MSE_bfgs_NPMLEinit_list.append(-100) 
                    train_SURE_bfgs_NPMLEinit_list.append(-100) 

                # misspec
                train_MSE_adam_misspec_list.append(adam_twonorm_diff_misspec/n)
                train_SURE_adam_misspec_list.append(adam_SURE_misspec[-1])
                train_MSE_bfgs_misspec_list.append(bfgs_twonorm_diff_misspec/n)
                train_SURE_bfgs_misspec_list.append(bfgs_SURE_misspec[-1])

                # TODO: Save model that has NA objective   
                

                ### Evaluate adam vs. bfgs on test data ###
                theta, Z, X = simulate_data.xie(experiment=experiment, n=n)
                sigma = X[:,-1]

                if count_exceptions < 5: 
                    test_adam_EB_NPMLEinit_thetahat = adam_model_NPMLEinit.get_theta_hat(n, B, Z, sigma)
                    test_bfgs_EB_NPMLEinit_thetahat = bfgs_model_NPMLEinit.get_theta_hat(n, B, Z, sigma)

                    test_MSE_adam_NPMLEinit_list.append(np.linalg.norm(test_adam_EB_NPMLEinit_thetahat - theta)**2 / n)
                    test_SURE_adam_NPMLEinit_list.append(adam_model_NPMLEinit.opt_func(Z, n, B, sigma).cpu().detach().numpy()) 

                    test_MSE_bfgs_NPMLEinit_list.append(np.linalg.norm(test_bfgs_EB_NPMLEinit_thetahat - theta)**2 / n)
                    test_SURE_bfgs_NPMLEinit_list.append(bfgs_model_NPMLEinit.opt_func(Z, n, B, sigma).cpu().detach().numpy()) 
                else:
                    test_MSE_adam_NPMLEinit_list.append(-100) 
                    test_SURE_adam_NPMLEinit_list.append(-100) 

                    test_MSE_bfgs_NPMLEinit_list.append(-100) 
                    test_SURE_bfgs_NPMLEinit_list.append(-100) 

                test_adam_EB_misspec_thetahat = adam_model_misspec.get_theta_hat(n, B, Z, sigma)
                test_bfgs_EB_misspec_thetahat = bfgs_model_misspec.get_theta_hat(n, B, Z, sigma)

                test_MSE_adam_misspec_list.append(np.linalg.norm(test_adam_EB_misspec_thetahat - theta)**2 / n)
                test_SURE_adam_misspec_list.append(adam_model_misspec.opt_func(Z, n, B, sigma).cpu().detach().numpy())
                test_MSE_bfgs_misspec_list.append(np.linalg.norm(test_bfgs_EB_misspec_thetahat - theta)**2 / n)
                test_SURE_bfgs_misspec_list.append(bfgs_model_misspec.opt_func(Z, n, B, sigma).cpu().detach().numpy())


            elif model=="wellspec":

                set_seed = rn.randint(low=1, high=2**32-1)
                # Train data
                theta, Z, X = simulate_data.xie(experiment=experiment, n=n)

                # EB - wellspecified

                ## Adam, ReLU default
                adam_result_wellspec = train.train_covariates(X, Z, theta, objective="SURE", set_seed=set_seed, d=2, B=100, drop_sigma=False)
                adam_model_wellspec, adam_Ft_Rep_wellspec, adam_SURE_wellspec, adam_NLL_wellspec, adam_score_wellspec, adam_theta_hats_wellspec, adam_twonorm_diff_wellspec = adam_result_wellspec
                # Add a plot of training curves?
                
                print("Finished training EB wellspecified with adam")

                ## BFGS and ELU
                ELU_result_wellspec = train.train_covariates(X, Z, theta, objective="SURE", set_seed=set_seed, d=2, B=100, drop_sigma=False,
                                                optimizer_str="bfgs", activation_fn=tr.nn.ELU())
                ELU_model_wellspec, ELU_Ft_Rep_wellspec, ELU_SURE_wellspec, ELU_NLL_wellspec, ELU_score_wellspec, ELU_theta_hats_wellspec, ELU_twonorm_diff_wellspec = ELU_result_wellspec
                # Add a plot of training curves?
                print("Finished training EB wellspecified with BFGS, ELU")

                # BFGS and SiLu
                SiLU_result_wellspec = train.train_covariates(X, Z, theta, objective="SURE", set_seed=set_seed, d=2, B=100, drop_sigma=False,
                                                optimizer_str='bfgs', activation_fn=tr.nn.SiLU())
                SiLU_model_wellspec, SiLU_Ft_Rep_wellspec, SiLU_SURE_wellspec, SiLU_NLL_wellspec, SiLU_score_wellspec, SiLU_theta_hats_wellspec, SiLU_twonorm_diff_wellspec = SiLU_result_wellspec
                # Add a plot of training curves?
                print("Finished training EB wellspecified with BFGS, SiLU")

                # Training losses
                train_adam_MSE_wellspec_list.append(adam_twonorm_diff_wellspec/n)
                train_ELU_MSE_wellspec_list.append(ELU_twonorm_diff_wellspec/n)
                train_SiLU_MSE_wellspec_list.append(SiLU_twonorm_diff_wellspec/n)

                train_adam_SURE_wellspec_list.append(adam_SURE_wellspec[-1])
                train_ELU_SURE_wellspec_list.append(ELU_SURE_wellspec[-1])
                train_SiLU_SURE_wellspec_list.append(SiLU_SURE_wellspec[-1])

                if (np.isnan(ELU_SURE_wellspec[-1]) and (not found_NA_ELU)):

                    path = "models/xie_failures/wellspec/" + experiment + "_ELU_bfgs" 

                    # Save objects after finding NA
                    np.save(path + "/theta", theta)
                    tr.save(Z, path +  "/Z")
                    tr.save(X, path + "/X")
                    
                    np.save(path + "/set_seed", set_seed)
                    tr.save(ELU_Ft_Rep_wellspec, path  + "/Ft_Rep")
                    tr.save(ELU_model_wellspec.state_dict(), path + "/ELU_model_wellspec")
                    tr.save(ELU_SURE_wellspec, path +  "/ELU_SURE_wellspec")

                    tr.save(SiLU_model_wellspec.state_dict(), path + "/SiLU_model_wellspec")
                    tr.save(SiLU_SURE_wellspec, path +  "/SiLU_SURE_wellspec")

                    tr.save(adam_model_wellspec.state_dict(), path + "/adam_model_wellspec")
                    tr.save(adam_SURE_wellspec, path +  "/adam_SURE_wellspec")

                    found_NA_ELU = True
                
                if (np.isnan(SiLU_SURE_wellspec[-1]) and (not found_NA_SiLU)):

                    path = "models/xie_failures/wellspec/" + experiment + "_SiLU_bfgs" 

                    # Save objects after finding NA
                    np.save(path + "/theta", theta)
                    tr.save(Z, path +  "/Z")
                    tr.save(X, path + "/X")
                    
                    np.save(path + "/set_seed", set_seed)
                    tr.save(ELU_Ft_Rep_wellspec, path + "/Ft_Rep")
                    tr.save(ELU_model_wellspec.state_dict(), path + "/ELU_model_wellspec")
                    tr.save(ELU_SURE_wellspec, path +  "/ELU_SURE_wellspec")

                    tr.save(SiLU_model_wellspec.state_dict(), path + "/SiLU_model_wellspec")
                    tr.save(SiLU_SURE_wellspec, path +  "/SiLU_SURE_wellspec")

                    tr.save(adam_model_wellspec.state_dict(), path + "/adam_model_wellspec")
                    tr.save(adam_SURE_wellspec, path +  "/adam_SURE_wellspec")

                    found_NA_SiLU = True
                
                ### Evaluate on test data ###
                theta, Z, X = simulate_data.xie(experiment=experiment, n=n)
                sigma = X[:,-1]


                # EB wellspec evaluation
                test_adam_wellspec_thetahat = adam_model_wellspec.get_theta_hat(n, Z, X)
                test_ELU_wellspec_thetahat = ELU_model_wellspec.get_theta_hat(n, Z, X)
                test_SiLU_wellspec_thetahat = SiLU_model_wellspec.get_theta_hat(n, Z, X)

                test_Ft_Rep = adam_model_wellspec.feature_representation(X) # the same for all activation functions
                test_adam_wellspec_SURE = adam_model_wellspec.opt_func_SURE(Z, test_Ft_Rep, X).cpu().detach().numpy()
                test_ELU_wellspec_SURE = ELU_model_wellspec.opt_func_SURE(Z, test_Ft_Rep, X).cpu().detach().numpy()
                test_SiLu_wellspec_SURE = SiLU_model_wellspec.opt_func_SURE(Z, test_Ft_Rep, X).cpu().detach().numpy()

                # Test losses
                test_adam_MSE_wellspec_list.append(np.linalg.norm(test_adam_wellspec_thetahat - theta)**2 / n)
                test_ELU_MSE_wellspec_list.append(np.linalg.norm(test_ELU_wellspec_thetahat - theta)**2 / n)
                test_SiLU_MSE_wellspec_list.append(np.linalg.norm(test_SiLU_wellspec_thetahat - theta)**2 / n)

                test_adam_SURE_wellspec_list.append(test_adam_wellspec_SURE)
                test_ELU_SURE_wellspec_list.append(test_ELU_wellspec_SURE)
                test_SiLU_SURE_wellspec_list.append(test_SiLu_wellspec_SURE)

            else:
                print("typo in model")
        
    # Save as df
    if model == "misspec":
        result = pd.DataFrame({
            'train_MSE_adam_misspec_list': train_MSE_adam_misspec_list,
            'train_MSE_adam_NPMLEinit_list': train_MSE_adam_NPMLEinit_list,
            'test_MSE_adam_misspec_list': test_MSE_adam_misspec_list,
            'test_MSE_adam_NPMLEinit_list': test_MSE_adam_NPMLEinit_list,
            'train_SURE_adam_misspec_list': train_SURE_adam_misspec_list,
            'train_SURE_adam_NPMLEinit_list': train_SURE_adam_NPMLEinit_list,
            'test_SURE_adam_misspec_list': test_SURE_adam_misspec_list,
            'test_SURE_adam_NPMLEinit_list': test_SURE_adam_NPMLEinit_list,
            'train_MSE_bfgs_misspec_list':  train_MSE_bfgs_misspec_list,
            'train_MSE_bfgs_NPMLEinit_list':  train_MSE_bfgs_NPMLEinit_list,
            'test_MSE_bfgs_misspec_list':  test_MSE_bfgs_misspec_list,
            'test_MSE_bfgs_NPMLEinit_list': test_MSE_bfgs_NPMLEinit_list,
            'train_SURE_bfgs_misspec_list': train_SURE_bfgs_misspec_list,
            'train_SURE_bfgs_NPMLEinit_list': train_SURE_bfgs_NPMLEinit_list,
            'test_SURE_bfgs_misspec_list': test_SURE_bfgs_misspec_list,
            'test_SURE_bfgs_NPMLEinit_list': test_SURE_bfgs_NPMLEinit_list,
            'experiment': experiments_list
        })

    elif model == "wellspec":
        result = pd.DataFrame({
            'train_adam_MSE_wellspec_list': train_adam_MSE_wellspec_list,
            'test_adam_MSE_wellspec_list':  test_adam_MSE_wellspec_list,
            'train_adam_SURE_wellspec_list': train_adam_SURE_wellspec_list,
            'test_adam_SURE_wellspec_list': test_adam_SURE_wellspec_list,
            'train_ELU_MSE_wellspec_list':  train_ELU_MSE_wellspec_list,
            'test_ELU_MSE_wellspec_list':  test_ELU_MSE_wellspec_list,
            'train_ELU_SURE_wellspec_list': train_ELU_SURE_wellspec_list,
            'test_ELU_SURE_wellspec_list': test_ELU_SURE_wellspec_list,
            'train_SiLU_MSE_wellspec_list':  train_SiLU_MSE_wellspec_list,
            'test_SiLU_MSE_wellspec_list': test_SiLU_MSE_wellspec_list,
            'train_SiLU_SURE_wellspec_list': train_SiLU_SURE_wellspec_list,
            'test_SiLU_SURE_wellspec_list': test_SiLU_SURE_wellspec_list,
            'experiment': experiments_list
        })


    else:
        print("typo in model")

    return result



def get_theta_hats(n, B, experiment, variance, endpoints=None):

    theta_hats = {}

    sigma = np.sqrt(variance)*tr.ones(n,)
    Z_new = simulate_data.xie_Z_grid(n, experiment, np.sqrt(variance),
                                     expanded=False, endpoints=endpoints) 
    X = sigma.reshape(n,1)
    # # xie_Z_grid takes SD, not variance

    # score_EB_misspec = globals()['model_EB_misspec' + experiment].compute_score(Z_new, n, B, sigma).detach().numpy()
    # score_EB_NPMLEinit = globals()['model_EB_NPMLEinit' + experiment].compute_score(Z_new, n, B, sigma).detach().numpy()
    # score_NPMLE = globals()['model_NPMLE' + experiment].compute_score(Z_new, n, B, sigma).detach().numpy()

    # sigma = np.sqrt(variance)*tr.ones(n,1)
    # Ft_Rep = globals()['model_EB_wellspec' + experiment].feature_representation(sigma)
    # score_EB_wellspec = globals()['model_EB_wellspec' + experiment].compute_score(Z_new, Ft_Rep, sigma).detach().numpy()

    # parametric
    # put theta as 0 because we don't care about MSE or SURE
    theta_hat_G, not_MSE, not_SURE, not_grand_mean, not_lambda_hat = models.theta_hat_G(0, Z_new, sigma)

    
    theta_hats['EB_misspec'] = globals()['model_EB_misspec' + experiment].get_theta_hat(n, B, Z_new, sigma)
    theta_hats['EB_NPMLEinit'] = globals()['model_EB_NPMLEinit' + experiment].get_theta_hat(n, B, Z_new, sigma)
    theta_hats['NPMLE'] = globals()['model_NPMLE' + experiment].get_theta_hat(n, B, Z_new, sigma)
    theta_hats['EB_wellspec'] = globals()['model_EB_wellspec' + experiment].get_theta_hat(n, Z_new, X)
    theta_hats['parametric_G'] = theta_hat_G
    if experiment == 'e': 
        if variance == 0.1: 
            theta_hats['truth'] = 0.5 * (Z_new + 2) 
        elif variance == 0.5: 
            theta_hats['truth'] = 0.5 * Z_new 
    else: 
        theta_hats['truth'] = variance 
    theta_hats['experiment'] = n*[experiment]
    theta_hats['variance'] = n*[variance]

    Z_new = Z_new.detach().numpy()
    theta_hats['Z'] = Z_new

    return(pd.DataFrame(theta_hats)) 

def xie_checks(n = 1000, B = 100, experiments = ['c', 'd', 'd5', 'e', 'f'], 
                  prior_dist_plot = True, SURE_loss_plot = True, wellspec_prior_plot = True, theta_hat_list_save = True, 
                  marginal_dist_plot = True, marginal_dist_expanded_plot = True, marginal_dist_avg_plot = True, 
                  ft_rep_string = ['Ft_Rep_wellspec'], data = ['X', 'Z', 'theta', 'grand_mean', 'lambda_hat'], 
                  model_states = ['pi_hat_NPMLE', 'model_EB_misspec', 'model_EB_NPMLEinit', 'model_EB_wellspec', 'model_NPMLE'], 
                  SURE_losses = ['SURE_misspec', 'SURE_NPMLEinit', 'SURE_wellspec', 'SURE_truth'], 
                  variance_dict = {"c": [0.2, 0.55, 0.9], "d": [0.01, 0.125, 1], "d5": [0.2, 0.7, 4.74, 10], "e": [0.1, 0.5], "f": [0.2, 0.55, 0.9]}, 
                  optimizer_str = "adam", activation_fn_str = "relu"): 
    
    """
    Save plots: prior, SURE training loss, theta hat dataframe, marginal, empirical margina
    """
    
    item_lists = [data, ft_rep_string, model_states, SURE_losses] 
    filename_end = "_" + optimizer_str + "_" + activation_fn_str + "_" + str(n)
 
    for experiment in experiments:

        print(experiment)

        path = "models/xie/" + experiment + "_" + optimizer_str + "_" + activation_fn_str + "/" 
        # filename_end = "_" + optimizer_str + "_" + activation_fn_str + filename_end

        for list in item_lists:
            for item_name in list:

                # print(item_name)

                if item_name == 'Ft_Rep_wellspec' or item_name == 'X' or item_name == 'Z' or 'SURE' in item_name : # tensor load
                    globals()[item_name + experiment] = tr.load(path + item_name,map_location=tr.device('cpu'), weights_only=True)
                elif 'model' not in item_name: # np load
                    globals()[item_name + experiment] = np.load(path + item_name + '.npy')
                # create model objects and load 
                elif item_name == 'model_EB_wellspec':
                    globals()[item_name + experiment] = models.model_covariates(X=globals()['X' + experiment], 
                                                                                Z=globals()['Z' + experiment],
                                                                                hidden_sizes=(256, 256),
                                                                                B=100)
                    globals()[item_name + experiment].load_state_dict(tr.load(path + item_name,map_location=tr.device('cpu'), weights_only=False))
                elif item_name == 'model_EB_misspec':
                    globals()[item_name + experiment] = models.model_theta_pi_sure(Z=globals()['Z' + experiment],
                                                                                B=B, 
                                                                                init_val_theta=tr.log(tr.Tensor([1.5])), 
                                                                                init_val_pi=tr.log(tr.Tensor([1.5])))
                    globals()[item_name + experiment].load_state_dict(tr.load(path + item_name,map_location=tr.device('cpu'), weights_only=True))
                elif item_name == 'model_NPMLE':
                    globals()[item_name + experiment] = models.model_pi_sure(Z=globals()['Z' + experiment],
                                                                                B=B, 
                                                                                init_val=tr.log(tr.tensor(globals()['pi_hat_NPMLE' + experiment])))
                elif item_name == 'model_EB_NPMLEinit':
                    globals()[item_name + experiment] = models.model_theta_pi_sure(Z=globals()['Z' + experiment],
                                                                                B=B, 
                                                                                init_val_theta=tr.log(tr.Tensor([1.5])), 
                                                                                init_val_pi=tr.log(tr.Tensor([1.5])))
                    globals()[item_name + experiment].load_state_dict(tr.load(path + item_name,map_location=tr.device('cpu'), weights_only=True))
    if prior_dist_plot == True: 

        prior_dict = {} # compute grids and priors

        for experiment in experiments:

            theta_grid_EB_misspec, pi_hat_EB_misspec = globals()['model_EB_misspec' + experiment].get_prior(globals()['Z' + experiment]) 

            theta_grid_EB_NPMLEinit, pi_hat_EB_NPMLEinit = globals()['model_EB_NPMLEinit' + experiment].get_prior(globals()['Z' + experiment]) 

            theta_grid_NPMLE, pi_hat_NPMLE = globals()['model_NPMLE' + experiment].get_prior(globals()['Z' + experiment]) 

            theta_grid_EB_wellspec, pi_hat_EB_wellspec = globals()['model_EB_wellspec' + experiment].get_prior(
                globals()['Z' + experiment],  globals()['Ft_Rep_wellspec' + experiment])

            prior_dict["theta_grid_EB_misspec" + experiment] = theta_grid_EB_misspec
            prior_dict["theta_grid_EB_NPMLEinit" + experiment] = theta_grid_EB_NPMLEinit
            prior_dict["theta_grid_NPMLE" + experiment] = theta_grid_NPMLE
            prior_dict["theta_grid_EB_wellspec" + experiment] = theta_grid_EB_wellspec

            prior_dict["pi_hat_EB_misspec" + experiment] = pi_hat_EB_misspec
            prior_dict["pi_hat_EB_NPMLEinit" + experiment] = pi_hat_EB_NPMLEinit
            prior_dict["pi_hat_NPMLE" + experiment] = pi_hat_NPMLE.detach().numpy()
            prior_dict["pi_hat_EB_wellspec" + experiment] = pi_hat_EB_wellspec
        
            fig, ax = plt.subplots(1, 3, figsize=(18,7))

            ax[0].stem(prior_dict["theta_grid_EB_misspec" + experiment], 
            prior_dict["pi_hat_EB_misspec" + experiment])
            ax[0].title.set_text('EB misspecified')

            ax[1].stem(prior_dict["theta_grid_EB_NPMLEinit" + experiment], 
            prior_dict["pi_hat_EB_NPMLEinit" + experiment])
            ax[1].title.set_text('EB NPMLE init')

            ax[2].stem(prior_dict["theta_grid_NPMLE" + experiment], 
            prior_dict["pi_hat_NPMLE" + experiment])
            ax[2].title.set_text('NPMLE')
            # ax.plot(xk, custm.pmf(xk), 'ro', ms=8, mec='r')
            # ax.vlines(xk, 0, custm.pmf(xk), colors='r', linestyles='-', lw=2)
            fig.suptitle("Experiment (" + experiment + ")")
            filename = "results/xie_checks/figures_priors/prior_dist_" + experiment + filename_end + ".png"
            plt.savefig(filename) 
            plt.close()
    
    if SURE_loss_plot == True: 

        for experiment in experiments: # plot

            plt.plot(globals()["SURE_misspec" + experiment], label = 'misspecified') 
            plt.plot(globals()["SURE_NPMLEinit" + experiment], label = 'misspecified - NPMLEinit') 
            plt.plot(globals()["SURE_wellspec" + experiment], label = 'wellspecified') 
            if experiment == 'e': 
                plt.axhline(globals()["SURE_truth" + experiment], label = 'truth', linestyle='--', color = 'r') 
                plt.axhline(0.15, label = 'Bayes risk', linestyle = '--', color = 'yellow')
            plt.legend()
            plt.title("SURE loss - Experiment (" + experiment + ")")
            filename = "results/xie_checks/figures_sure_losses/SURE_loss_" + experiment + filename_end + ".png"
            plt.savefig(filename) 
            plt.close()

    if wellspec_prior_plot == True: 

        for experiment in experiments:

            values, sorted_idx = globals()["X" + experiment].reshape(n,).sort()

            fig, (ax1, ax2) = plt.subplots(1,2, figsize = (13, 5))
            sns.heatmap(prior_dict["pi_hat_EB_wellspec" + experiment][sorted_idx], ax=ax1)
            sns.heatmap(prior_dict["theta_grid_EB_wellspec" + experiment][sorted_idx], ax=ax2)
            ax1.title.set_text("pi hat")
            ax2.title.set_text("theta grid")
            plt.suptitle("Wellspecified prior from experiment (" + experiment + ")")
            filename = "results/xie_checks/figures_priors/wellspec_prior_" + experiment + filename_end + ".png"
            plt.savefig(filename) 
            plt.close()
    
    if theta_hat_list_save == True: 

        theta_hat_list = []

        for experiment, variance_list in variance_dict.items():
            for variance in variance_list:
                theta_hat_list.append(get_theta_hats(n, B, experiment, variance)) 
        
        theta_hat_df = pd.concat(theta_hat_list)
        theta_hat_df.to_csv('results/xie_checks/xie_shrinkage' + filename_end + '.csv') 
    
    if marginal_dist_expanded_plot == True: 

        for experiment, variance_list in variance_dict.items():

            number_columns_to_plot = len(variance_list)

            fig, ax = plt.subplots(1, number_columns_to_plot, figsize=(18,7))

            Z_experiment = globals()["Z" + experiment] 

            for column in range(number_columns_to_plot):

                variance = variance_list[column]
                sigma_float = np.sqrt(variance)

                # if experiment == "d5":

                    # Z_grid_incomplete = simulate_data.xie_Z_grid(n, experiment, sigma_float, expanded=True)

                    # fifth_idx = tr.tensor(np.linspace(0, 995, 200), dtype=int)
                    # other_idx = np.setdiff1d(range(n), np.linspace(0, 995, 200))
                    # Z_grid = tr.cat((Z_grid[other_idx],Z_grid[fifth_idx]*3))
                    # Z_grid, idx = Z_grid.sort()
                    
                # else: 
                    # Z_grid = simulate_data.xie_Z_grid(n, experiment, sigma_float, expanded=True)
                
                Z_grid = tr.tensor(np.linspace(min(Z_experiment).detach().numpy(), max(Z_experiment).detach().numpy(), n)) 

                X_fixed = sigma_float*tr.ones(n,1)

                grand_mean = globals()["grand_mean" + experiment].item()
                lambda_hat = globals()["lambda_hat" + experiment].item()

                ax[column].plot(Z_grid,
                                globals()["model_EB_misspec" + experiment].get_marginal(n, B, Z_grid, sigma_float),
                                color="red", label='EB misspecified')

                ax[column].plot(Z_grid,
                                globals()["model_EB_NPMLEinit" + experiment].get_marginal(n, B, Z_grid, sigma_float),
                                color="blue", linestyle = "dashed", label='EB misspecific NPMLE init')

                ax[column].plot(Z_grid,
                                globals()["model_NPMLE" + experiment].get_marginal(n, B, Z_grid, sigma_float),
                                color="green", label='NPMLE')

                ax[column].plot(Z_grid, globals()["model_EB_wellspec" + experiment].get_marginal(Z_grid, X_fixed),
                                color="orange", label='EB wellspecified')

                ax[column].plot(Z_grid, ss.norm.pdf(Z_grid, loc=grand_mean, scale=np.sqrt(lambda_hat)),
                                color="purple", label='parametric')
                
                if experiment != "e":
                    ax[column].plot(Z_grid, ss.norm.pdf(Z_grid, loc=variance, scale=np.sqrt(variance)),
                                    color="gray", label='truth', linestyle="dashed")
                else:
                    
                    if sigma_float == np.sqrt(0.1):
                        ax[column].plot(Z_grid, ss.norm.pdf(Z_grid, loc=2, scale=np.sqrt(2*variance)),
                                        color="gray", label='truth', linestyle="dashed")
                    else:
                        ax[column].plot(Z_grid, ss.norm.pdf(Z_grid, loc=0, scale=np.sqrt(2*variance)),
                                        color="gray", label='truth', linestyle="dashed")
                
                ax[column].set_title("Variance = " + str(variance))    

            fig.suptitle("Marginal of experiment (" + experiment + ")")
            fig.tight_layout() 
            plt.legend(loc="upper center", ncol=number_columns_to_plot, bbox_to_anchor=(0.05, -0.05))
            filename = "results/xie_checks/figures_marginals/xie_expanded_marginal_" + experiment + filename_end + ".png"
            plt.savefig(filename) 
            plt.close()

    if marginal_dist_plot == True: 

        for experiment, variance_list in variance_dict.items():

            number_columns_to_plot = len(variance_list)

            fig, ax = plt.subplots(1, number_columns_to_plot, figsize=(18,7))

            Z_experiment = globals()["Z" + experiment] 

            for column in range(number_columns_to_plot):

                variance = variance_list[column]
                sigma_float = np.sqrt(variance)

                # if experiment == "d5":

                    # Z_grid_incomplete = simulate_data.xie_Z_grid(n, experiment, sigma_float, expanded=False)

                    # fifth_idx = tr.tensor(np.linspace(0, 995, 200), dtype=int)
                    # other_idx = np.setdiff1d(range(n), np.linspace(0, 995, 200))
                    # Z_grid = tr.cat((Z_grid[other_idx],Z_grid[fifth_idx]*3))
                    # Z_grid, idx = Z_grid.sort()
                    
                # else: 
                    # Z_grid = simulate_data.xie_Z_grid(n, experiment, sigma_float, expanded=False)

                Z_grid = tr.tensor(np.linspace(min(Z_experiment).detach().numpy(), max(Z_experiment).detach().numpy(), n)) 

                X_fixed = sigma_float*tr.ones(n,1)

                grand_mean = globals()["grand_mean" + experiment].item()
                lambda_hat = globals()["lambda_hat" + experiment].item()

                ax[column].plot(Z_grid,
                                globals()["model_EB_misspec" + experiment].get_marginal(n, B, Z_grid, sigma_float),
                                color="red", label='EB misspecified')

                ax[column].plot(Z_grid,
                                globals()["model_EB_NPMLEinit" + experiment].get_marginal(n, B, Z_grid, sigma_float),
                                color="blue", linestyle = "dashed", label='EB misspecific NPMLE init')

                ax[column].plot(Z_grid,
                                globals()["model_NPMLE" + experiment].get_marginal(n, B, Z_grid, sigma_float),
                                color="green", label='NPMLE')

                ax[column].plot(Z_grid, globals()["model_EB_wellspec" + experiment].get_marginal(Z_grid, X_fixed),
                                color="orange", label='EB wellspecified')

                ax[column].plot(Z_grid, ss.norm.pdf(Z_grid, loc=grand_mean, scale=np.sqrt(lambda_hat)),
                                color="purple", label='parametric')
                
                if experiment != "e":
                    ax[column].plot(Z_grid, ss.norm.pdf(Z_grid, loc=variance, scale=np.sqrt(variance)),
                                    color="gray", label='truth', linestyle="dashed")
                else:

                    if sigma_float == np.sqrt(0.1):
                        ax[column].plot(Z_grid, ss.norm.pdf(Z_grid, loc=2, scale=np.sqrt(2*variance)),
                                        color="gray", label='truth', linestyle="dashed")
                    else:
                        ax[column].plot(Z_grid, ss.norm.pdf(Z_grid, loc=0, scale=np.sqrt(2*variance)),
                                        color="gray", label='truth', linestyle="dashed")
                
                ax[column].set_title("Variance = " + str(variance))    

            fig.suptitle("Marginal of experiment (" + experiment + ")")
            fig.tight_layout() 
            plt.legend(loc="upper center", ncol=number_columns_to_plot, bbox_to_anchor=(0.05, -0.05))
            filename = "results/xie_checks/figures_marginals/xie_marginal_" + experiment + filename_end + ".png"
            plt.savefig(filename) 
            plt.close()
    
    if marginal_dist_avg_plot == True: 

        for experiment in experiments:

            number_columns_to_plot = 1 

            fig, ax = plt.subplots(1, number_columns_to_plot, figsize=(18,7))

            sigma_experiment = globals()["X" + experiment] 
            Z_experiment = globals()["Z" + experiment] 

            Z_grid = tr.tensor(np.linspace(min(Z_experiment).detach().numpy(), max(Z_experiment).detach().numpy(), n)) 

            model_EB_misspec_marginal_avg = np.zeros((n,)) 
            model_EB_NPMLEinit_marginal_avg = np.zeros((n,)) 
            model_NPMLE_marginal_avg = np.zeros((n,))
            model_EB_wellspec_marginal_avg = np.zeros((n,)) 
            parametric_marginal_avg = np.zeros((n,)) 
            truth_marginal_avg = np.zeros((n,)) 

            for column in range(len(sigma_experiment)):

                # variance = variance_list[column]
                sigma_float = sigma_experiment[column] 
                variance = sigma_float**2 

                # if experiment == "d5":

                    # Z_grid_incomplete = simulate_data.xie_Z_grid(n, experiment, sigma_float, expanded=False)

                    # fifth_idx = tr.tensor(np.linspace(0, 995, 200), dtype=int)
                    # other_idx = np.setdiff1d(range(n), np.linspace(0, 995, 200))
                    # Z_grid = tr.cat((Z_grid[other_idx],Z_grid[fifth_idx]*3))
                    # Z_grid, idx = Z_grid.sort()
                    
                # else: 
                    # Z_grid = simulate_data.xie_Z_grid(n, experiment, sigma_float, expanded=False)

                X_fixed = sigma_float*tr.ones(n,1)

                grand_mean = globals()["grand_mean" + experiment].item()
                lambda_hat = globals()["lambda_hat" + experiment].item()

                model_EB_misspec_marginal_avg += globals()["model_EB_misspec" + experiment].get_marginal(n, B, Z_grid, sigma_float)
                model_EB_NPMLEinit_marginal_avg += globals()["model_EB_NPMLEinit" + experiment].get_marginal(n, B, Z_grid, sigma_float)
                model_NPMLE_marginal_avg += globals()["model_NPMLE" + experiment].get_marginal(n, B, Z_grid, sigma_float)
                model_EB_wellspec_marginal_avg += globals()["model_EB_wellspec" + experiment].get_marginal(Z_grid, X_fixed)
                parametric_marginal_avg += ss.norm.pdf(Z_grid, loc=grand_mean, scale=np.sqrt(lambda_hat)) 

                if experiment != "e":
                    truth_marginal_avg += ss.norm.pdf(Z_grid, loc=variance, scale=np.sqrt(variance)) 
                
                else:
                    if sigma_float == np.sqrt(0.1):
                        truth_marginal_avg += ss.norm.pdf(Z_grid, loc=2, scale=np.sqrt(2*variance)) 
                    elif sigma_float == np.sqrt(0.5):
                        truth_marginal_avg += ss.norm.pdf(Z_grid, loc=0, scale=np.sqrt(2*variance)) 
                    else: 
                        print("Error occurred in experiment e")

            ax.plot(Z_grid, model_EB_misspec_marginal_avg/n, color="red", label='EB misspecified')

            ax.plot(Z_grid, model_EB_NPMLEinit_marginal_avg/n, color="blue", linestyle = "dashed", label='EB misspecific NPMLE init')

            ax.plot(Z_grid, model_NPMLE_marginal_avg/n, color="green", label='NPMLE')

            ax.plot(Z_grid, model_EB_wellspec_marginal_avg/n, color="orange", label='EB wellspecified')

            ax.plot(Z_grid, parametric_marginal_avg/n, color="purple", label='parametric')

            ax.plot(Z_grid, truth_marginal_avg/n, color="gray", label='truth', linestyle="dashed")
            
            # ax.set_title("Variance = " + str(variance))    

            fig.suptitle("Average marginal of experiment (" + experiment + ")")
            fig.tight_layout() 
            plt.legend(loc="upper center", ncol=number_columns_to_plot, bbox_to_anchor=(0.05, -0.05))
            filename = "results/xie_checks/figures_marginals/xie_marginal_average_" + experiment + filename_end + ".png"
            plt.savefig(filename) 
            plt.close() 


def read_location_scale(seeds,
                        B=100, experiments = ['c', 'd', 'd5', 'e', 'f', 'g'], 
                        ft_rep_string = ['Ft_Rep_wellspec', 'Ft_Rep_surels'], data = ['X', 'Z', 'theta', 'grand_mean', 'lambda_hat'], 
                        model_states = ['pi_hat_NPMLE', 'model_EB_misspec', 'model_EB_NPMLEinit', 'model_EB_wellspec', 'model_NPMLE', 'model_EB_surels'], 
                        SURE_losses = ['SURE_misspec', 'SURE_NPMLEinit', 'SURE_wellspec', 'SURE_truth', 'SURE_surels'], 
                        simulate_location_list=[True, False],
                        simulate_scale_list=[True, False], path="models/xie_location_scale"):
    
    item_lists = [data, ft_rep_string, model_states, SURE_losses] 
    path = path + "_" + "".join(str(seed) for seed in seeds) + "/"

    data_dict = {}
    parameteric_model_dict = {}
    NPMLE_model_dict = {}
    wellspec_model_dict = {}
    misspec_model_dict = {}
    NPMLEinit_model_dict = {}
    surels_model_dict = {}

    # Read models
    print("Reading data and models")
    for experiment in experiments:

        experiment_str = "_" + experiment 
        print(experiment)

        for list in item_lists:
            for item_name in list:

                # Read items that don't depend on use_location or use_scale
                if item_name == 'X' or item_name == 'Z' or item_name == 'SURE_truth': # tensor load
                    data_dict[item_name + experiment_str] = tr.load(path + experiment + "/" + item_name,
                                                                    map_location=tr.device('cpu'), weights_only=True)
                elif item_name == "pi_hat_NPMLE":
                    NPMLE_model_dict[item_name + experiment_str] = np.load(path + experiment + "/" + item_name + ".npy")
                elif 'model' not in item_name and 'SURE' not in item_name and "Ft_Rep" not in item_name: # np load prametric, 
                    parameteric_model_dict[item_name + experiment_str] = np.load(path + experiment + "/" + item_name + ".npy")
                elif item_name == 'model_NPMLE':
                    NPMLE_model_dict[item_name + experiment_str] = models.model_pi_sure_no_grid_modeling(Z=data_dict['Z' + experiment_str],
                                                                                B=B, 
                                                                                init_val=tr.log(tr.tensor(NPMLE_model_dict['pi_hat_NPMLE' + experiment_str])))
                elif item_name == 'SURE_surels' or item_name == 'Ft_Rep_surels': 
                    surels_model_dict[item_name + experiment_str] = tr.load(path + experiment + "/" + item_name,map_location=tr.device('cpu'), weights_only=True)
                elif item_name == 'model_EB_surels': 
                    surels_model_dict[item_name + experiment_str] = models.model_sure_ls(X=data_dict['X' + experiment_str], 
                                                                                         Z=data_dict['Z' + experiment_str], d=2, 
                                                                                         hidden_sizes=(8, 8)) 
                    surels_model_dict[item_name + experiment_str].load_state_dict(tr.load(path + experiment + "/" + item_name, 
                                                                                          map_location=tr.device('cpu'), 
                                                                                          weights_only=False))
                # Read items that *do* depend on the suffix, eg. _location_scale
                else: 
                    for use_scale in simulate_scale_list:
                        for use_location in simulate_location_list:

                            if use_location & use_scale:
                                suffix = "_location_scale"
                            elif use_location & (not use_scale): 
                                suffix = "_location_IQR"
                            elif (not use_location) & use_scale:
                                suffix = "_median_scale"
                            else:
                                suffix = "_median_IQR"
                    
                            if ('SURE' in item_name and item_name != 'SURE_surels') or item_name == 'Ft_Rep_wellspec':
                                wellspec_model_dict[item_name + experiment_str + suffix] = tr.load(path + experiment + "/" + item_name + suffix,map_location=tr.device('cpu'), weights_only=True)
                            elif item_name == 'model_EB_wellspec':
                                wellspec_model_dict[item_name + experiment_str + suffix] = models.model_covariates(X=data_dict['X' + experiment_str], 
                                                                                            Z=data_dict['Z' + experiment_str],
                                                                                            hidden_sizes=(8, 8),
                                                                                            B=100, use_location=use_location,
                                                                                                    use_scale=use_scale)
                                wellspec_model_dict[item_name + experiment_str + suffix].load_state_dict(tr.load(path + experiment + "/" + item_name + suffix, 
                                                                                                    map_location=tr.device('cpu'), 
                                                                                                    weights_only=False))
                            elif item_name == 'model_EB_misspec':
                                misspec_model_dict[item_name + experiment_str + suffix] = models.model_theta_pi_sure(Z=data_dict['Z' + experiment_str],
                                                                                            B=B, 
                                                                                            init_val_theta=tr.log(tr.Tensor([1.5])), 
                                                                                            init_val_pi=tr.log(tr.Tensor([1.5])), use_location=use_location,
                                                                                                    use_scale=use_scale)
                                misspec_model_dict[item_name + experiment_str + suffix].load_state_dict(tr.load(path + experiment + "/" + item_name + suffix, 
                                                                                                    map_location=tr.device('cpu'), 
                                                                                                    weights_only=True))
                            elif item_name == 'model_EB_NPMLEinit':
                                NPMLEinit_model_dict[item_name + experiment_str + suffix] = models.model_theta_pi_sure(Z=data_dict['Z' + experiment_str],
                                                                                            B=B, 
                                                                                            init_val_theta=tr.log(tr.Tensor([1.5])), 
                                                                                            init_val_pi=tr.log(tr.Tensor([1.5])))
                                NPMLEinit_model_dict[item_name + experiment_str + suffix].load_state_dict(tr.load(path + experiment + "/" + item_name  + suffix, 
                                                                                                    map_location=tr.device('cpu'), 
                                                                                                weights_only=True))
                                
    return(data_dict, parameteric_model_dict, NPMLE_model_dict, wellspec_model_dict, misspec_model_dict, NPMLEinit_model_dict, surels_model_dict)


def save_theta_hats(seeds, 
                    data_dict, parameteric_model_dict, NPMLE_model_dict, wellspec_model_dict, misspec_model_dict, NPMLEinit_model_dict, surels_model_dict,
                    n=6400, B=100,
                    variance_dict = {"c": [0.2, 0.55, 0.9], "d": [0.01, 0.125, 1], "d5": [0.2, 0.7, 4.74, 10], "e": [0.1, 0.5], "f": [0.2, 0.55, 0.9], "g": [0.2, 0.35, 0.5]},
                    simulate_location_list=[True, False],
                    simulate_scale_list=[True, False], save_path="results/xie_checks",
                    endpoints=None): 
    
    """
    Save plots: prior, SURE training loss, theta hat dataframe, marginal, empirical margina
    """

    save_path = save_path + "_" + "".join(str(seed) for seed in seeds) + "/"
    theta_hat_list = []

    for experiment, variance_list in variance_dict.items():
        experiment_str = "_" + experiment 

        print(experiment)

        for variance in variance_list:

            theta_hats = {}
            sigma = np.sqrt(variance)*tr.ones(n,)

            Z_new = simulate_data.xie_Z_grid(n, experiment, np.sqrt(variance), endpoints=endpoints) 

            X = sigma.reshape(n,1)
            
            theta_hats['NPMLE'] = NPMLE_model_dict['model_NPMLE' + experiment_str].get_theta_hat(n, B, Z_new, sigma)

            grand_mean = parameteric_model_dict["grand_mean" + experiment_str] 
            lambda_hat = parameteric_model_dict["lambda_hat" + experiment_str]
            theta_hats['parametric_G'] = (lambda_hat / (variance + lambda_hat)) * Z_new.detach().numpy() + (variance / (variance + lambda_hat)) * grand_mean

            if experiment == 'e': 
                if variance == 0.1: 
                    theta_hats['truth'] = 0.5 * (Z_new + 2) 
                elif variance == 0.5: 
                    theta_hats['truth'] = 0.5 * Z_new 
            else: 
                theta_hats['truth'] = variance 
            theta_hats['experiment'] = n*[experiment]
            theta_hats['variance'] = n*[variance]

            for use_scale in simulate_scale_list:
                for use_location in simulate_location_list:

                    if use_location & use_scale:
                        suffix = "_location_scale"
                    elif use_location & (not use_scale): 
                        suffix = "_location_IQR"
                    elif (not use_location) & use_scale:
                        suffix = "_median_scale"
                    else:
                        suffix = "_median_IQR"

                    theta_hats['EB_misspec'  + suffix ] = misspec_model_dict['model_EB_misspec' + experiment_str + suffix ].get_theta_hat(n, B, Z_new, sigma)
                    theta_hats['EB_NPMLEinit' + suffix] = NPMLEinit_model_dict['model_EB_NPMLEinit' + experiment_str + suffix].get_theta_hat(n, B, Z_new, sigma)
                    theta_hats['EB_wellspec' + suffix] = wellspec_model_dict['model_EB_wellspec' + experiment_str + suffix].get_theta_hat(n, Z_new, X)
                    
            theta_hats['EB_surels'] = surels_model_dict['model_EB_surels' + experiment_str].get_theta_hat(Z_new, X).detach().numpy() 

            Z_new = Z_new.detach().numpy()
            theta_hats['Z'] = Z_new

            theta_hat_list.append(pd.DataFrame(theta_hats)) 
    
    theta_hat_df = pd.concat(theta_hat_list)
    theta_hat_df.to_csv(save_path + 'xie_shrinkage_location_scale.csv') 


def save_marginals_data_location_scale(seeds,
                                       parametric_model_dict, NPMLE_model_dict, wellspec_model_dict, misspec_model_dict, NPMLEinit_model_dict, surels_model_dict,
                                       n=6400, B=100, 
                                       covariate_dict = {"j": [0, 0.5, 1]},
                                       variance_dict = {"c": [0.2, 0.55, 0.9], "d": [0.01, 0.125, 1], "d5": [0.2, 0.7, 4.74, 10], 
                                                        "e": [0.1, 0.5], "f": [0.2, 0.55, 0.9], "g": [0.2, 0.35, 0.45], 
                                                        "j": [1, 4, 8]},
                                       simulate_location_list=[True, False],
                                       simulate_scale_list=[True, False], expanded=False, save_path="results/xie_checks"):
    
    save_path = save_path + "_" + "".join(str(seed) for seed in seeds) + "/"

    experiment_list = []
    variance_for_df_list = []
    use_location_list = []
    use_scale_list = []

    Z_grid_list = []

    NPMLE_marginal_list = []
    parametric_marginal_list = []
    misspec_marginal_list = []
    NPMLEinit_marginal_list = []
    wellspec_marginal_list = []
    surels_marginal_list = []
    true_marginal_list = []


    
    for experiment, variance_list in variance_dict.items():
        
        experiment_str = "_" + experiment 
        number_columns_to_plot = len(variance_list)

        for use_scale in simulate_scale_list:
            for use_location in simulate_location_list:

                if use_location & use_scale:
                    suffix = "_location_scale"
                elif use_location & (not use_scale): 
                    suffix = "_location_IQR"
                elif (not use_location) & use_scale:
                    suffix = "_median_scale"
                else:
                    suffix = "_median_IQR"


                for column in range(number_columns_to_plot):

                    variance = variance_list[column]
                    sigma_float = np.sqrt(variance)

                    if expanded:
                        Z_grid = simulate_data.xie_Z_grid(n, experiment, sigma_float, expanded=expanded)
                    else:
                        Z_grid = np.linspace(wellspec_model_dict["model_EB_wellspec" + experiment_str + suffix].min_Z.detach().numpy(), 
                                                        wellspec_model_dict["model_EB_wellspec" + experiment_str + suffix].max_Z.detach().numpy(), n)
                        Z_grid = tr.tensor(Z_grid)

                    experiment_list.extend(n*[experiment])
                    variance_for_df_list.extend(n*[variance])
                    use_location_list.extend(n*[use_location])
                    use_scale_list.extend(n*[use_scale])
                    Z_grid_list.extend(Z_grid.detach().numpy().tolist())

                    # Z_grid = tr.tensor(Z_grid)
                    if experiment != "j": 
                        X_fixed = sigma_float*tr.ones(n,1)
                    else: 
                        X_wo_sigma = covariate_dict[experiment][column]*tr.ones(n,1)
                        X_sigma = sigma_float*tr.ones(n,1)
                        X_fixed = tr.cat((X_wo_sigma, X_sigma), dim=1) 

                    grand_mean = parametric_model_dict["grand_mean" + experiment_str].item()
                    lambda_hat = parametric_model_dict["lambda_hat" + experiment_str].item()

                    NPMLE_marginal = NPMLE_model_dict["model_NPMLE" + experiment_str].get_marginal(n, B, Z_grid, sigma_float).tolist()
                    parametric_marginal = ss.norm.pdf(Z_grid, loc=grand_mean, scale=np.sqrt(lambda_hat)).tolist()
                    misspec_marginal = misspec_model_dict["model_EB_misspec" + experiment_str + suffix].get_marginal(n, B, Z_grid, sigma_float).tolist()
                    NPMLEinit_marginal = NPMLEinit_model_dict["model_EB_NPMLEinit" + experiment_str + suffix].get_marginal(n, B, Z_grid, sigma_float).tolist()
                    wellspec_marginal = wellspec_model_dict["model_EB_wellspec" + experiment_str + suffix].get_marginal(Z_grid, X_fixed).tolist()
                    
                    Ft_Rep_surels = surels_model_dict["model_EB_surels" + experiment_str].feature_representation(X_fixed)
                    lambda_surels, b_surels = surels_model_dict["model_EB_surels" + experiment_str].forward(Ft_Rep_surels) 
                    A_surels = ((sigma_float**2) * (1-lambda_surels) / lambda_surels).detach().numpy()
                    m_surels = (b_surels / lambda_surels).detach().numpy()
                    surels_marginal = ss.norm.pdf(Z_grid, loc=m_surels, scale=np.sqrt(sigma_float**2 + A_surels)).tolist() 

                    
                    if experiment != "e" and experiment != "g": 
                        true_marginal = ss.norm.pdf(Z_grid, loc=variance, scale=np.sqrt(variance)).tolist()
                    elif experiment == "e":
                        if sigma_float == np.sqrt(0.1):
                            true_marginal = ss.norm.pdf(Z_grid, loc=2, scale=np.sqrt(2*variance)).tolist()
                        else:
                            true_marginal = ss.norm.pdf(Z_grid, loc=0, scale=np.sqrt(2*variance)).tolist()
                    else: 
                        true_marginal = (0.5*ss.norm.pdf(Z_grid, loc=variance, scale=np.sqrt(variance)) + 0.5*ss.norm.pdf(Z_grid, loc=20*variance, scale=np.sqrt(variance))).tolist()

                    NPMLE_marginal_list.extend(NPMLE_marginal)
                    parametric_marginal_list.extend(parametric_marginal)
                    misspec_marginal_list.extend(misspec_marginal)
                    NPMLEinit_marginal_list.extend(NPMLEinit_marginal)
                    wellspec_marginal_list.extend(wellspec_marginal)
                    surels_marginal_list.extend(surels_marginal)
                    true_marginal_list.extend(true_marginal)

    df = pd.DataFrame({'experiment': experiment_list,
                              'variance': variance_for_df_list,
                              'use_location': use_location_list,
                              'use_scale': use_scale_list,
                              'Z': Z_grid_list,
                              'NPMLE': NPMLE_marginal_list,
                              'thetaG': parametric_marginal_list,
                              'misspec': misspec_marginal_list,
                              'NPMLEinit': NPMLEinit_marginal_list,
                              'wellspec': wellspec_marginal_list,
                              'surels': surels_marginal_list, 
                              'truth': true_marginal_list})

    if expanded:
        df.to_csv(save_path + "location_scale_marginals_expanded" + experiment_str + ".csv")
    else:
        df.to_csv(save_path + "location_scale_marginals" + experiment_str + ".csv")

def save_empirical_marginals_data_location_scale(seeds,
                                                 data_dict, parametric_model_dict, NPMLE_model_dict, wellspec_model_dict, misspec_model_dict, NPMLEinit_model_dict,
                                                 experiments = ['c', 'd', 'd5', 'e', 'f'], 
                                       n=6400, B=100, 
                                       simulate_location_list=[True, False],
                                       simulate_scale_list=[True, False], save_path="results/xie_checks"):
    
    save_path = save_path + "_" + "".join(str(seed) for seed in seeds) + "/"

    experiment_list = []
    use_location_list = []
    use_scale_list = []

    Z_grid_list = []

    NPMLE_empirical_marginal_list = []
    parametric_empirical_marginal_list = []
    misspec_empirical_marginal_list = []
    NPMLEinit_empirical_marginal_list = []
    wellspec_empirical_marginal_list = []
    true_empirical_marginal_list = []


    for experiment in experiments:

            experiment_str = "_" + experiment 

            X_train = data_dict["X" + experiment_str] 
            Z_train = data_dict["Z" + experiment_str] 

            Z_grid_np = np.linspace(min(Z_train).detach().numpy(), max(Z_train).detach().numpy(), n)
            Z_grid = tr.tensor(Z_grid_np)

            for use_scale in simulate_scale_list:
                for use_location in simulate_location_list:

                    if use_location & use_scale:
                        suffix = "_location_scale"
                    elif use_location & (not use_scale): 
                        suffix = "_location_IQR"
                    elif (not use_location) & use_scale:
                        suffix = "_median_scale"
                    else:
                        suffix = "_median_IQR"

                    experiment_list.extend(n*[experiment])
                    use_location_list.extend(n*[use_location])
                    use_scale_list.extend(n*[use_scale])
                    Z_grid_list.extend(Z_grid_np.tolist())

                    misspec_empirical_marginal = np.zeros((n,)) 
                    NPMLEinit_empirical_marginal = np.zeros((n,)) 
                    NPMLE_empirical_marginal = np.zeros((n,))
                    wellspec_empirical_marginal = np.zeros((n,)) 
                    parametric_empirical_marginal = np.zeros((n,)) 
                    true_empirical_marginal = np.zeros((n,)) 

                    for idx in range(len(X_train)):

                        # variance = variance_list[column]
                        sigma_float = X_train[idx] 
                        variance = sigma_float**2
                        sigma_n1 = sigma_float*tr.ones(n,1)


                        grand_mean = parametric_model_dict["grand_mean" + experiment_str].item()
                        lambda_hat = parametric_model_dict["lambda_hat" + experiment_str].item()

                        parametric_empirical_marginal += ss.norm.pdf(Z_grid, loc=grand_mean, scale=np.sqrt(lambda_hat)) 
                        NPMLE_empirical_marginal += NPMLE_model_dict["model_NPMLE" + experiment_str].get_marginal(n, B, Z_grid, sigma_float)

                        misspec_empirical_marginal += misspec_model_dict["model_EB_misspec" + experiment_str + suffix].get_marginal(n, B, Z_grid, sigma_float)
                        NPMLEinit_empirical_marginal += NPMLEinit_model_dict["model_EB_NPMLEinit" + experiment_str + suffix].get_marginal(n, B, Z_grid, sigma_float)
                        wellspec_empirical_marginal += wellspec_model_dict["model_EB_wellspec" + experiment_str + suffix].get_marginal(Z_grid, sigma_n1)

                        if experiment != "e":
                            true_empirical_marginal += ss.norm.pdf(Z_grid, loc=variance, scale=np.sqrt(variance))
                        
                        else:
                            if sigma_float == np.sqrt(0.1):
                                true_empirical_marginal += ss.norm.pdf(Z_grid, loc=2, scale=np.sqrt(2*variance)) 
                            elif sigma_float == np.sqrt(0.5):
                                true_empirical_marginal += ss.norm.pdf(Z_grid, loc=0, scale=np.sqrt(2*variance)) 
                            else: 
                                print("Error occurred in experiment e")
                    
                    # did not divide by n 
                    NPMLE_empirical_marginal_list.extend(NPMLE_empirical_marginal.tolist())
                    parametric_empirical_marginal_list.extend(parametric_empirical_marginal.tolist())
                    misspec_empirical_marginal_list.extend(misspec_empirical_marginal.tolist())
                    NPMLEinit_empirical_marginal_list.extend(NPMLEinit_empirical_marginal.tolist())
                    wellspec_empirical_marginal_list.extend(wellspec_empirical_marginal.tolist())
                    true_empirical_marginal_list.extend(true_empirical_marginal.tolist())


    df = pd.DataFrame({'experiment': experiment_list,
                              'use_location': use_location_list,
                              'use_scale': use_scale_list,
                              'Z': Z_grid_list,
                              'NPMLE': NPMLE_empirical_marginal_list,
                              'thetaG': parametric_empirical_marginal_list,
                              'misspec': misspec_empirical_marginal_list,
                              'NPMLEinit': NPMLEinit_empirical_marginal_list,
                              'wellspec': wellspec_empirical_marginal_list,
                              'truth': true_empirical_marginal_list})

    df.to_csv(save_path + "location_scale_empirical_marginals.csv")

def save_priors_location_scale(seeds,
                               data_dict, NPMLE_model_dict, misspec_model_dict, NPMLEinit_model_dict, 
                               experiments = ['c', 'd', 'd5', 'e', 'f'], 
                               simulate_location_list=[True, False],
                               simulate_scale_list=[True, False], save_path="results/xie_checks", B=100):
    
    save_path = save_path + "_" + "".join(str(seed) for seed in seeds) + "/"

    prior_dict = {} # compute grids and priors


    for experiment in experiments:
        experiment_str = "_" + experiment 

        for use_scale in simulate_scale_list:
            for use_location in simulate_location_list:

                if use_location & use_scale:
                    suffix = "_location_scale"
                elif use_location & (not use_scale): 
                    suffix = "_location_IQR"
                elif (not use_location) & use_scale:
                    suffix = "_median_scale"
                else:
                    suffix = "_median_IQR"

                min_Z = min(data_dict['Z' + experiment_str].detach().numpy())
                max_Z  = max(data_dict['Z' + experiment_str].detach().numpy())
                n = data_dict['Z' + experiment_str].detach().numpy().shape[0]
                Z_grid_np = np.linspace(min_Z, max_Z, n)
                Z = tr.tensor(Z_grid_np)
                


                theta_grid_NPMLE, pi_hat_NPMLE = NPMLE_model_dict['model_NPMLE' + experiment_str].get_prior(Z) 

                theta_grid_EB_misspec, pi_hat_EB_misspec = misspec_model_dict['model_EB_misspec' + experiment_str + suffix].get_prior(n, B, Z) 
                theta_grid_EB_NPMLEinit, pi_hat_EB_NPMLEinit = NPMLEinit_model_dict['model_EB_NPMLEinit' + experiment_str + suffix].get_prior(n, B, Z) 
                # theta_grid_EB_wellspec, pi_hat_EB_wellspec = globals()['model_EB_wellspec' + experiment_str + suffix].get_prior(
                #     Z,  globals()['Ft_Rep_wellspec' + experiment_str + suffix])
                # Ft_Rep_surels = globals()['Ft_Rep_surels' + experiment_str] 
                # X = globals()['X' + experiment_str] 
                # lambda_surels, b_surels = surels_model_dict["model_EB_surels" + experiment_str].forward(Ft_Rep_surels) 
                # A_surels = (X**2) * (1-lambda_surels) / lambda_surels
                # m_surels = b_surels / lambda_surels

                prior_dict["theta_grid_EB_misspec" + experiment_str + suffix] = theta_grid_EB_misspec
                prior_dict["theta_grid_EB_NPMLEinit" + experiment_str + suffix] = theta_grid_EB_NPMLEinit
                prior_dict["theta_grid_NPMLE" + experiment_str + suffix] = theta_grid_NPMLE
                # prior_dict["theta_grid_EB_wellspec" + experiment_str + suffix] = theta_grid_EB_wellspec
                # prior_dict["m_surels" + experiment_str] = m_surels

                prior_dict["pi_hat_EB_misspec" + experiment_str + suffix] = pi_hat_EB_misspec
                prior_dict["pi_hat_EB_NPMLEinit" + experiment_str + suffix] = pi_hat_EB_NPMLEinit
                prior_dict["pi_hat_NPMLE" + experiment_str + suffix] = pi_hat_NPMLE.detach().numpy()
                # prior_dict["pi_hat_EB_wellspec" + experiment_str + suffix] = pi_hat_EB_wellspec
                # prior_dict["A_surels" + experiment_str] = A_surels 
                    
    df = pd.DataFrame(prior_dict)

    df.to_csv(save_path + "location_scale_priors.csv")
