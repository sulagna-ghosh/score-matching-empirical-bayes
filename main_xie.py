# Executes checks for eight different experiments conducted to get figure 2 and 3 in section 6.2.1 

import experiments_xie
import pandas as pd
import torch as tr 

import sys  
print(sys.path)

############## Save one run of Xie models: location and scale ###########

seeds = [600, 50, 100] # for final plot
experiments = ['c', 'd', 'e', 'f']
variance_dict = {"c": [0.2, 0.55, 0.9], "d": [0.01, 0.125, 1], "e": [0.1, 0.5], "f": [0.2, 0.55, 0.9]}
experiments_xie.train_and_save_models_location_scale(seeds=seeds, experiments=experiments, 
                                                     simulate_location_list=[False],simulate_scale_list=[True])


############## Xie checks (priors, marginals, shrinkage rules): location and scale ##############

data_dict, parameteric_model_dict, NPMLE_model_dict, wellspec_model_dict, misspec_model_dict, NPMLEinit_model_dict, surels_model_dict = experiments_xie.read_location_scale(seeds=seeds,
                                                                                                                                                        experiments=experiments, 
                                                                                                                                                        simulate_location_list=[False],
                                                                                                                                                        simulate_scale_list=[True])

print("Begin to save theta hats")
experiments_xie.save_theta_hats(seeds, data_dict, parameteric_model_dict, NPMLE_model_dict, wellspec_model_dict, 
                                misspec_model_dict, NPMLEinit_model_dict, surels_model_dict, 
                                n=6400, variance_dict = variance_dict,
                                endpoints = [-4, 4])

print("Begin to save marginals")
experiments_xie.save_marginals_data_location_scale(seeds,
                                                   parameteric_model_dict, NPMLE_model_dict, wellspec_model_dict, 
                                                   misspec_model_dict, NPMLEinit_model_dict, surels_model_dict, expanded=False,
                                                   variance_dict=variance_dict, 
                                                   simulate_location_list=[False],  simulate_scale_list=[True]) 

print("Begin to save priors")
experiments_xie.save_priors_location_scale(seeds,
                                           data_dict, NPMLE_model_dict, misspec_model_dict, NPMLEinit_model_dict,
                                           experiments=experiments) 