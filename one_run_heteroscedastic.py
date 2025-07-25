# Executes checks for eight different experiments conducted to get figure 2 and 3 in section 6.2.1 

import experiment_heteroscedastic
import pandas as pd
import torch as tr 

import sys  
print(sys.path)

############## Save one run of Xie models: location and scale ###########

seeds = [600, 50, 100] # for final plot
experiments = ['e']
variance_dict = {"e": [0.1, 0.5]}
# variance_dict = {"c": [0.2, 0.55, 0.9], "d": [0.01, 0.125, 1], "e": [0.1, 0.5], "f": [0.2, 0.55, 0.9]}
# experiment_heteroscedastic.train_and_save_models_location_scale(seeds=seeds, experiments=experiments, 
#                                                      simulate_location_list=[False],simulate_scale_list=[True])


############## Xie checks (priors, marginals, shrinkage rules): location and scale ##############

data_dict, parameteric_model_dict, NPMLE_model_dict, wellspec_model_dict, misspec_model_dict, NPMLEinit_model_dict, surels_model_dict = experiment_heteroscedastic.read_location_scale(seeds=seeds,
                                                                                                                                                        experiments=experiments, 
                                                                                                                                                        simulate_location_list=[False],
                                                                                                                                                        simulate_scale_list=[True])

print("Begin to save theta hats")
experiment_heteroscedastic.save_theta_hats(seeds, data_dict, parameteric_model_dict, NPMLE_model_dict, wellspec_model_dict, 
                                misspec_model_dict, NPMLEinit_model_dict, surels_model_dict, 
                                n=6400, variance_dict = variance_dict,
                                endpoints = [-4, 4], 
                                simulate_location_list=[False],  simulate_scale_list=[True])

print("Begin to save marginals")
experiment_heteroscedastic.save_marginals_data_location_scale(seeds,
                                                   parameteric_model_dict, NPMLE_model_dict, wellspec_model_dict, 
                                                   misspec_model_dict, NPMLEinit_model_dict, surels_model_dict, expanded=False,
                                                   variance_dict=variance_dict, 
                                                   simulate_location_list=[False],  simulate_scale_list=[True]) 

print("Begin to save priors")
experiment_heteroscedastic.save_priors_location_scale(seeds,
                                           data_dict, NPMLE_model_dict, misspec_model_dict, NPMLEinit_model_dict,
                                           experiments=experiments, 
                                           simulate_location_list=[False],  simulate_scale_list=[True]) 