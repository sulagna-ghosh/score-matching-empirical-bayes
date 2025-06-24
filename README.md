# Score matching empirical bayes 
This is a repository of the codes used to get results for the paper "Stein's unbiased risk estimate and Hyvarinen's score matching". Note to remember for this repository, 'wellspec' means SURE-THING, 'misspec' means SURE-PM, '...G' means SURE-grandmean and 'ls' for SURE-LS. 

## Folders 
1. models: Contains saved models for SURE-PM, SURE-THING, NPMLE and SURE-grandmean the first four experiments from xie et.al. 2012 in section 6.2, such as, 
    - xie: models from experiments c, d, e, f with different optimizers and activation functions; 
    - xie_location_scale_*: models from experiments c, d, e, f with adam optimizer and relu activation function with a specific combination of starting seeds, denoted as *. 
2. results: Contains specific .csv files along with .png files, used to run the results and get what we need, such as, 
    - ICSDS: Specific images for use and understand specific cases in section 6; 
    - no_covariates: All results involving SURE-PM in section 6.1; 
    - scores: Results to check scores from different models and experiments; 
    - xie_checks_*: Checks to verify the xie experiments are working as expected for our models, when we use a specific combination of starting seeds, denoted as *, using marginals, priors and shrinkages; 
    - xie_losses: Contains all results with models (run with location and scale parameters), and tested for 
        1. figures_xie_losses: Figures we get from the tests done; 
        2. final_results: Contains all .csv files to get figure 1 in section 6.2; 
        3. hidden_layer_same_run_results: Contains .csv files to compare several different hidden layer sizes and check if the use of skip connections is helping; 
    - xie_plots_with_all: Contains all necessaru images used in section 6; 
    - plot_results_final.R: R file used to generate all the plots in section 6; 
    - plot_results_preliminary.R: R file used to test different comparisons mentioned above. 

## .py files 
1. bayes_risk_calculation: Numerical calculation of bayes risk for experiments with no closed form bayes risk; 
2. experiment_covariates_highD.py and main_covariates.py: Codes for comparing the models which use covariates (including SURE-THING); 
3. experiment_nocovariates.py, main_nocovariates.py and submitit_nocovariates.py: Codes for comparing the models with no covariates (including SURE-PM), which gives us the tables in section 6.1; 
4. experiments_xie.py, main_xie_checks and submitit_xie.py: All codes involving the experiments done in section 6.2 with eight different experiments and different models; 
5. main_optimizers.py: Codes to compare effect of different optimizers and activation functions; 
6. models.py: Contains all the models (SURE-PM, SURE-THING, SURE-grandmean, SURE-LS and EBCF); 
7. score.py: Codes to test for score values for models with the truth; 
8. simulate_data.py: Codes to simulate data in different scenarios in section 6; 
9. train.py: Functions to train different models (SURE-PM, SURE-THING, SURE-grandmean, SURE-LS, NPMLE and EBCF). 
