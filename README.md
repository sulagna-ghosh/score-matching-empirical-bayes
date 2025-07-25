# Replication code for "Stein's unbiased risk estimate and Hyvarinen's score matching". 

This repository contains the implementation of various EB estimators and code for experiments in sections 5-7 of the paper "Stein's unbiased risk estimate and Hyvarinen's score matching". 

The following is an outline of sections 5-7 and the corresponding files in the repository.

* Section 5: Computational strategy for SURE-training
    - `models.py` contains classes that correspond to SURE-PM, SURE-THING, SURE-LS, and other benchmark estimators.
    - `train.py` contains functions that train the estimators in `models.py`. 

* Section 6: Numerical results
* Section 6.1: Homoscedastic setting without side-information
    - The folder `results/homoscedastic` contains the simulation results for the *homosecedastic* experiments *without covariates*. Files named `normal_homo_*.csv` are the results for the normal-normal setting (Table 1); files named `binary_homo_*.csv` are the results for the binary $\mu$ setting (Table 2).
    - The R file `plot_results_homoscedastic.R` processes the above .csv files.
    - `submitit_homoscedastic.py` generates the .csv files by running the simulations on SLURM. The file has dependencies: `experiment_homoscedastic.py`, `train.py`, `models.py`, and `simulate_data.py`.

* Section 6.2: Heteroscedastic setting
    - The folder `results/heteroscedastic` contains the simulation results for the *heteroscedastic* experiments *with covariates*. Files named `location_scale_comparison_*.csv` contain the in- and out-sample MSE of various estimators for experiments (c) through (g). (These simulation results include the four specifications of fixing or training the location $m$ and scale $s$ parameters.) 
    - The R file `plot_results_heteroscedastic.R` processes the above .csv files to create Figure 1.
    - `submitit_heteroscedastic.py` generates the .csv files by running the simulations on SLURM. The file has dependencies: `experiment_heteroscedastic.py`, `train.py`, `models.py`, and `simulate_data.py`.
    - `miscellaneous/bayes_risk_calculation.ipynb` contains numerical calculations of the Bayes risks for the experiments with no closed form Bayes risk. 

* Section 6.2.1: Comparison of SURE-PM and NPMLE in the bimodal case
    - The folder `results/one_run_heteroscedastic` contains .csv files for the prior, marginal, and shrinkage rule of *one run* of the bimodal heterscedastic (experiment e). The folder also contains the saved models themselves (that were created from `one_run_heteroscedastic.py`).
    - The R file `plot_results_heteroscedastic.R` processes the above .csv files to create Figures 2 and 3.
    - The final figures are in the folder `results/figures`.

* Section 7: Application to the Opportunity Atlas 



Note to remember for this repository, 'wellspec' means SURE-THING, 'misspec' means SURE-PM, '...G' means SURE-grandmean and 'ls' for SURE-LS. Finally, `requirements.txt` contains all the libraries of Python and R that will be needed to replicate the results. 

