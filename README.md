# Replication code for "Stein's unbiased risk estimate and Hyvarinen's score matching". 

This repository contains the implementation of various EB estimators and code for experiments in sections 5-7 of the paper "Stein's unbiased risk estimate and Hyvarinen's score matching". 

The following is an outline of sections 5-7 and the corresponding files in the repository.

* Section 5: Computational strategy for SURE-training
    - `models.py` contains classes that correspond to SURE-PM, SURE-THING, SURE-LS, and other benchmark estimators.
    - `train.py` contains functions that train the estimators in `models.py`. 

* Section 6: Numerical results
* Section 6.1: Homoscedastic setting without side-information
    - The folder `results/homoscedastic` contains the simulation results for the *homosecedastic* experiments *without covariates*. Files named `normal_homo_*.csv` are the results for the normal-normal setting (Table 1); files named `binary_homo_*.csv` are the results for the binary $\mu$ setting (Table 2).
    - The R file `plot_results_nocovariates.R` processes the above .csv files.
    - `submitit_homoscedastic.py` generates the .csv files by running the simulations on SLURM. The file has dependencies: `experiment_homoscedastic.py`, `train.py`, `models.py`, and `simulate_data.py`.

* Section 6.2: Heteroscedastic setting
    - The folder `results/heteroscedastic` contains the simulation results for the *heteroscedastic* experiments *with covariates*. Files named `location_scale_comparison_*.csv` contain the in- and out-sample MSE of various estimators for experiments (c) through (g). (These simulation results include the four specifications of fixing or training the location $m$ and scale $s$ parameters.) 
    - The R file `plot_results_heteroscedastic.R` processes the above .csv files to create Figure 1.
    - `submitit_covariates.py` generates the .csv files by running the simulations on SLURM. The file has dependencies: `experiment_heteroscedastic.py`, `train.py`, `models.py`, and `simulate_data.py`.

* Section 6.2.1: Comparison of SURE-PM and NPMLE in the bimodal case
    - The folder `results/one_run_heteroscedastic` contains .csv files for the prior, marginal, and shrinkage rule of *one run* of the bimodal heterscedastic (experiment c). The folder also contains the saved models themselves (that were created from `one_run_heteroscedastic.py`).
    - The R file `plot_results_heteroscedastic.R` processes the above .csv files to create Figures 2 and 3.
    - The final figures are in the folder `results/figures`.

* Section 7: Application to the Opportunity Atlas 



Note to remember for this repository, 'wellspec' means SURE-THING, 'misspec' means SURE-PM, '...G' means SURE-grandmean and 'ls' for SURE-LS. 




## Folders 
1. results: Contains specific .csv files along with .png files, used to run the results and get what we need, such as, 
    - no_covariates: All results involving SURE-PM for getting Table 1 and 2 in section 6.1, such as: 
        - binary_homo_*.csv: csv files containing simulations for homoscedastic no covariates case with binary mu, providing Table 2 in section 6.1; 
        - normal_homo_*.csv: csv files containing simulations for homoscedastic no covariates case with binary mu, providing Table 1 in section 6.1; 
    - xie_losses: Contains all .csv files to get figure 1, 2 and 3 in section 6.2, such as: 
        - location_scale_comparison_*.csv: csv files containing simulations for different heteroscedastic cases with covariates, providing Figure 1 in section 6.2; 
        - location_scale_priors.csv: contains priors from a single simulation for experiment with bimodal mu, two-point sigma, providing Figure 2 in section 6.2; 
        - location_scale_marginals.csv and xie_shrinkage_location_scale.csv: contain mu_hat's and marginals from a single simulation for experiment with bimodal mu, two-point sigma, providing Figure 3 in section 6.2; 
    - xie_plots: Contains Figures 1, 2 and 3 in section 6.2. 

## .py files 
1. simulate_data.py: Codes to simulate data in different scenarios in section 6; 
2. models.py: Contains all the models (SURE-PM, SURE-THING, SURE-LS, EBCF and SURE-grandmean) defined and used for getting the results; 
3. train.py: Functions to train all the models (SURE-PM, SURE-THING, SURE-grandmean, SURE-LS, NPMLE and EBCF); 
4. experiment_nocovariates.py: Function to generate dataframes for binary or normal mu and homoscedastic Z for section 6.1; 
5. main_nocovariates.py and submitit_nocovariates.py: Both files contain functions to save the dataframes for binary or normal mu and homoscedastic Z for section 6.1 (either of them works to get results, second one is for the case if you run it in cluster); 
6. plot_results_nocovariates.R: Codes for getting Table 1 and 2 in section 6.1; 
7. bayes_risk_calculation: Numerical calculation of bayes risk for xie experiments in section 6.2 with no closed form bayes risk; 
8. experiments_xie.py: Function to generate dataframes for different heteroscedastic cases with covariates for section 6.2; 
9. main_xie.py: Contain functions to save results from a single replicate for heteroscedastic experiments with covariates for Figure 2 and 3 in section 6.2; 
10. submitit_xie.py: Contain functions to save the dataframes for different heteroscedastic cases with covariates for Figure 1 in section 6.2; 
11. plot_results_xie.R: Codes for getting images for eight different heteroscedastic experiments with different models, which give Figure 1, 2 and 3 in section 6.2; 
12. requirements.txt: Contains all the libraries of Python and R that will be needed to replicate the results. 
