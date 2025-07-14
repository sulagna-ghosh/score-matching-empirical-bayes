# Executes different experiments conducted to get table 1 and 2 in section 6.1 

import submitit


def no_covariates_binary_simulation(m_sim=20):
    
    print("Start location_scale_comparison")

    n = experiment_nocovariates.n 
    homoskedastic_sigma = experiment_nocovariates.homoskedastic_sigma
    problems_binary_theta = experiment_nocovariates.problems_binary_theta
    
    results_binary_homoeskedastic_case = experiment_nocovariates.make_df_binary(problems_binary_theta, sigma=homoskedastic_sigma, m_sim=m_sim) 

    csv_names = glob.glob("results/no_covariates/binary_homo_*.csv")
    if len(csv_names) == 0:
        results_binary_homoeskedastic_case.to_csv('results/no_covariates/binary_homo_0.csv', index=False) 
    else:

        underscore_max_idx_list = [max([pos for pos, char in enumerate(list_item) if char == "_"]) for list_item in csv_names]
        period_max_idx_list  = [max([pos for pos, char in enumerate(list_item) if char == "."]) for list_item in csv_names]
        unique_suffix = max([int(csv_names[idx][(underscore_max_idx_list[idx]+1):period_max_idx_list[idx]]) for idx in range(len(csv_names))]) + 1 # add one to make it unique
        
        filename = "results/no_covariates/binary_homo_" + str(unique_suffix) + ".csv"
        results_binary_homoeskedastic_case.to_csv(filename) 
        

def no_covariates_normal_simulation(m_sim=20):
    
    print("Start location_scale_comparison")

    n = experiment_nocovariates.n
    homoskedastic_sigma = experiment_nocovariates.homoskedastic_sigma
    problems_normal_theta = experiment_nocovariates.problems_normal_theta
    
    results_normal_homoeskedastic_case = experiment_nocovariates.make_df_normal(problems_normal_theta, sigma=homoskedastic_sigma, m_sim=m_sim) 

    csv_names = glob.glob("results/no_covariates/normal_homo_*.csv")
    if len(csv_names) == 0:
        results_normal_homoeskedastic_case.to_csv('results/no_covariates/normal_homo_0.csv', index=False) 
    else:

        underscore_max_idx_list = [max([pos for pos, char in enumerate(list_item) if char == "_"]) for list_item in csv_names]
        period_max_idx_list  = [max([pos for pos, char in enumerate(list_item) if char == "."]) for list_item in csv_names]
        unique_suffix = max([int(csv_names[idx][(underscore_max_idx_list[idx]+1):period_max_idx_list[idx]]) for idx in range(len(csv_names))]) + 1 # add one to make it unique
        
        filename = "results/no_covariates/normal_homo_" + str(unique_suffix) + ".csv"
        results_normal_homoeskedastic_case.to_csv(filename) 
    


if __name__ == "__main__":
    import experiment_nocovariates
    import pandas as pd
    import glob

    print("%j")
    log_folder="submitit_log/%j"

    executor = submitit.AutoExecutor(folder=log_folder)

    executor.update_parameters(name="EB", slurm_partition="general", gpus_per_node=1, nodes=5,
                               mem_gb=3, timeout_min=600)

    job = executor.submit(no_covariates_normal_simulation, m_sim=5) 
    # job = executor.submit(no_covariates_binary_simulation, m_sim=5) 

    print(f"job.job_id: {job.job_id}")

