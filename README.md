# Score matching empirical bayes 
This is a repository of the codes used to get results for the paper "Stein's unbiased risk estimate and Hyvarinen's score matching". Note to remember for this repository, 'wellspec' means SURE-THING, 'misspec' means SURE-PM, '...G' means SURE-grandmean and 'ls' for SURE-LS. 

## Folders 
1. results: Contains specific .csv files along with .png files, used to run the results and get what we need, such as, 
    - no_covariates: All results involving SURE-PM for getting Table 1 and 2 in section 6.1; 
    - xie_losses: Contains all .csv files to get figure 1, 2 and 3 in section 6.2; 
    - xie_plots: Contains all necessary images used in section 6.2. 

## .py files 
1. simulate_data.py: Codes to simulate data in different scenarios in section 6; 
2. models.py: Contains all the models (SURE-PM, SURE-THING, SURE-LS, EBCF and SURE-grandmean); 
3. train.py: Functions to train different models (SURE-PM, SURE-THING, SURE-grandmean, SURE-LS, NPMLE and EBCF); 
4. experiment_nocovariates.py, main_nocovariates.py, submitit_nocovariates.py and plot_results_nocovariates.R: Codes for comparing the models with no covariates (including SURE-PM), which give Table 1 and 2 in section 6.1; 
5. bayes_risk_calculation: Numerical calculation of bayes risk for xie experiments in Section 6.2 with no closed form bayes risk; 
6. experiments_xie.py, main_xie_checks, submitit_xie.py and plot_results_xie.R: All codes involving the experiments done involving covariates with eight different experiments and different models, which give Figure 1, 2 and 3 in Section 6.2. 