# Executes different experiments conducted to get table 1 and 2 in section 6.1 

import submitit
from experiment_homoscedastic import n, homoskedastic_sigma
from experiment_homoscedastic import problems_binary_theta, make_discrete_homoscedastic_df
from experiment_homoscedastic import problems_normal_theta, make_normal_homoscedastic_df


def get_unique_suffix(csv_names):

    underscore_max_idx_list = [max([pos for pos, char in enumerate(list_item) if char == "_"]) for list_item in csv_names]
    period_max_idx_list  = [max([pos for pos, char in enumerate(list_item) if char == "."]) for list_item in csv_names]
    unique_suffix = max([int(csv_names[idx][(underscore_max_idx_list[idx]+1):period_max_idx_list[idx]]) for idx in range(len(csv_names))]) + 1 # add one to make it unique

    return unique_suffix

def save_discrete_df(m_sim):
    print(f"n: {n}\n")
    
    df = make_discrete_homoscedastic_df(problems_binary_theta, sigma=homoskedastic_sigma, m_sim=m_sim) 

    # Save df
    csv_names = glob.glob("results/homoscedastic/binary_homo_*.csv")
    if len(csv_names) == 0:
        df.to_csv('results/homoscedastic/binary_homo_0.csv', index=False) 
    else:
        unique_suffix = get_unique_suffix(csv_names)
        filename = "results/homoscedastic/binary_homo_" + str(unique_suffix) + ".csv"
        df.to_csv(filename) 
        
def save_normal_df(m_sim):
    
    df = make_normal_homoscedastic_df(problems_normal_theta, sigma=homoskedastic_sigma, m_sim=m_sim) 

    # Save df
    csv_names = glob.glob("results/homoscedastic/normal_homo_*.csv")
    if len(csv_names) == 0:
        df.to_csv('results/homoscedastic/normal_homo_0.csv', index=False) 
    else:
        unique_suffix = get_unique_suffix(csv_names)
        filename = "results/homoscedastic/normal_homo_" + str(unique_suffix) + ".csv"
        df.to_csv(filename) 
    


if __name__ == "__main__":
    import experiment_homoscedastic
    import pandas as pd
    import glob

    print("%j")
    log_folder="submitit_log/%j"

    executor = submitit.AutoExecutor(folder=log_folder)

    executor.update_parameters(name="EB", slurm_partition="general", gpus_per_node=1, nodes=2,
                               mem_gb=3, timeout_min=600)

    # job = executor.submit(save_normal_df, m_sim=25) 
    job = executor.submit(save_discrete_df, m_sim=25) 

    print(f"job.job_id: {job.job_id}")

