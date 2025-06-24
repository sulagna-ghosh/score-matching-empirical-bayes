import experiments_xie
import pandas as pd
import torch as tr 

import sys  
print(sys.path)

############## Save one run of Xie models: location and scale ###########

seeds = [600, 50, 100] # for final plot
experiments = ['j']
# experiments = ['c', 'd', 'e', 'f']
covariate_dict = {"j": [0, 0.5, 1]}
variance_dict = {"j": [1, 4, 8]}
# variance_dict = {"c": [0.2, 0.55, 0.9], "d": [0.01, 0.125, 1], "e": [0.1, 0.5], "f": [0.2, 0.55, 0.9]}
experiments_xie.train_and_save_models_location_scale(seeds=seeds, experiments=experiments, 
                                                     simulate_location_list=[False],simulate_scale_list=[True])

# seeds = [374, 55, 999] # 
# experiments_xie.train_and_save_models_location_scale(seeds=seeds, experiments=experiments,
#                                                      simulate_location_list=[False],simulate_scale_list=[True])


############## Xie checks (priors, marginals, shrinkage rules): location and scale ##############

for seeds in [[600, 50, 100]]:

    data_dict, parameteric_model_dict, NPMLE_model_dict, wellspec_model_dict, misspec_model_dict, NPMLEinit_model_dict, surels_model_dict = experiments_xie.read_location_scale(seeds=seeds,
                                                                                                                                                            experiments=experiments, 
                                                                                                                                                            simulate_location_list=[False],
                                                                                                                                                            simulate_scale_list=[True])

    print("Begin to save theta hats")
    # experiments_xie.save_theta_hats(seeds, data_dict, parameteric_model_dict, NPMLE_model_dict, wellspec_model_dict, 
    #                                 misspec_model_dict, NPMLEinit_model_dict, surels_model_dict, 
    #                                 n=6400, variance_dict = variance_dict,
    #                                 # endpoints=[-2.325998, 3.0307898])
    #                                 endpoints = [-4, 4])

    print("Begin to save marginals")
    # experiments_xie.save_marginals_data_location_scale(seeds,
    #                                                 parameteric_model_dict, NPMLE_model_dict, wellspec_model_dict, misspec_model_dict, NPMLEinit_model_dict, surels_model_dict, expanded=True,
    #                                                 variance_dict=variance_dict, covariate_dict=covariate_dict, simulate_location_list=[False],  simulate_scale_list=[True])
    # experiments_xie.save_marginals_data_location_scale(seeds,
    #                                                 parameteric_model_dict, NPMLE_model_dict, wellspec_model_dict, misspec_model_dict, NPMLEinit_model_dict, surels_model_dict, expanded=False,
    #                                                 variance_dict=variance_dict, covariate_dict=covariate_dict, simulate_location_list=[False],  simulate_scale_list=[True])

    print("Begin to save priors")
    # experiments_xie.save_priors_location_scale(seeds,
    #                                         data_dict, NPMLE_model_dict, misspec_model_dict, NPMLEinit_model_dict,
    #                                         experiments=experiments)

    # print("Begin to save empirical marginals")
    # experiments_xie.save_empirical_marginals_data_location_scale(seeds,
    #                                                             data_dict, parameteric_model_dict, NPMLE_model_dict, wellspec_model_dict, misspec_model_dict, NPMLEinit_model_dict,
    #                                                             experiments=experiments)



############## No seeds ###########

############## Save one run of Xie models ###########


# experiments_xie.train_and_save_models(n=10000, optimizer_str="bfgs", activation_fn_str="elu", experiments = ['e'])
# experiments_xie.train_and_save_models(n=1000, experiments = ['e'])
# experiments_xie.train_and_save_models(n=5000, experiments = ['c', 'd', 'd5', 'e', 'f'], optimizer_str="bfgs", verbose=True)

############## Xie checks (priors, marginals, shrinkage rules) ##############

# experiments_xie.xie_checks(n = 10000, experiments = ['e'], optimizer_str = "adam", activation_fn_str="relu", variance_dict = {"e": [0.1, 0.5]}) 
# experiments_xie.xie_checks(n = 1000, experiments = ['e'], optimizer_str = "bfgs", activation_fn_str="silu", variance_dict = {"e": [0.1, 0.5]}) 
# experiments_xie.xie_checks(n = 1000, experiments = ['e'], filename_end = "_1000") 

# Save NA-SURE models (debugging) ###########

# for model_fails_str in ["wellspec"]:

    # print("Searching for failure in " + model_fails_str)

    # experiments_xie.train_and_save_models_failures(n=10000, experiments = ["d5"],
                                        # optimizer_str="bfgs", model_fails = model_fails_str,
                                        # verbose=True)