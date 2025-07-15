import simulate_data
import train
import models

import numpy as np
import numpy.random as rn
import pandas as pd
import torch as tr 
import scipy.stats as ss
import random

# experiments c)-f) in xie 2012 and four additional experiments in section 6.2 

def simulate_location_scale(ns=[100, 200, 400, 800, 1600, 3200, 6400],
                          experiments=["c", "d", "d5", "e", "f"],
                          hidden_sizes=(8,8), B=100,
                          optimizer_str="adam",
                          simulate_location_list=[True],
                          simulate_scale_list=[True], 
                          skip_connect=False):
    '''
    Function that gives results for the eight different experiment simulations with different models. This is the 
    final function used to get results of figure 1 in section 6.2. This also takes into account if location and 
    scale will be used in our model. 
    '''

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
            hidden_sizes=(8,8), 
            simulate_location_list=[True],
            simulate_scale_list=[True], 
            skip_connect=False):
    """
    Compute MSEs and SURE on train and test for all models, m_sim times and returns a concatenated dataframe. 
    """

    mse_sure_results = [] # list of dataframes
    
    for m in range(m_sim):
        print(f"m_sim: {m}")
        mse_sure_results.append(simulate_location_scale(ns=ns, experiments=experiments,
                                                        hidden_sizes=hidden_sizes, B=B,
                                                        optimizer_str=optimizer_str,
                                                        simulate_location_list=simulate_location_list,
                                                        simulate_scale_list=simulate_scale_list, 
                                                        skip_connect=skip_connect)) 

    mse_sure_df = pd.concat(mse_sure_results)

    return mse_sure_df 

def train_and_save_models_location_scale(seeds,
                                         n=6400, 
                                         hidden_sizes=(8,8),
                                         B=100,
                                         experiments = ["c", "d", "d5", "e", "f", "g"],
                                         simulate_location_list=[True, False],
                                         simulate_scale_list=[True, False],
                                         optimizer_str="adam", path="models/xie_location_scale"):
    """
    Trains and saves the models to be used later. Does include the possibility of location and scale use in our model. 

    * optimizer_str: describes the optimizer used. for optimizers that AREN'T "adam",
                     save the model output in a <experiment>_<optimizer_str>
    """

    path = path + "_" + ''.join(str(seed) for seed in seeds) + "/"

    for experiment in experiments:

        print(f"experiment: {experiment}")
        
        experiment_str = experiment 
        
        print(experiment_str)

        rn.seed(seeds[0])
        random.seed(seeds[1])
        tr.manual_seed(seeds[2])

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

def read_location_scale(seeds,
                        B=100, experiments = ['c', 'd', 'd5', 'e', 'f', 'g'], 
                        ft_rep_string = ['Ft_Rep_wellspec', 'Ft_Rep_surels'], data = ['X', 'Z', 'theta', 'grand_mean', 'lambda_hat'], 
                        model_states = ['pi_hat_NPMLE', 'model_EB_misspec', 'model_EB_NPMLEinit', 'model_EB_wellspec', 'model_NPMLE', 'model_EB_surels'], 
                        SURE_losses = ['SURE_misspec', 'SURE_NPMLEinit', 'SURE_wellspec', 'SURE_truth', 'SURE_surels'], 
                        simulate_location_list=[True, False],
                        simulate_scale_list=[True, False], path="models/xie_location_scale"):
    
    '''
    Reads the models to be used later. Does include the possibility of location and scale use in our model. 
    '''
    
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
    
    """
    Save marginals from the models with loaction and scale. 
    """

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
        df.to_csv(save_path + "location_scale_marginals_expanded.csv")
    else:
        df.to_csv(save_path + "location_scale_marginals.csv")

def save_priors_location_scale(seeds,
                               data_dict, NPMLE_model_dict, misspec_model_dict, NPMLEinit_model_dict, 
                               experiments = ['c', 'd', 'd5', 'e', 'f'], 
                               simulate_location_list=[True, False],
                               simulate_scale_list=[True, False], save_path="results/xie_checks", B=100):
    
    """
    Save priors from the models with loaction and scale. 
    """
    
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
