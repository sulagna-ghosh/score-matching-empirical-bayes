# Executes eight different experiments conducted to get figure 1 in section 6.2 

import submitit


def location_scale_comparison(m_sim, ns=[100, 200, 400, 800, 1600, 3200, 6400],
                          experiments=["c", "d", "d5", "e", "f"],
                          hidden_sizes=(8,8), 
                          optimizer_str="adam",
                          simulate_location_list=[True, False],
                          simulate_scale_list=[True, False], 
                          skip_connect=False):
    
    print("Start location_scale_comparison")

    df = experiment_heteroscedastic.make_df(m_sim=m_sim,
                                 ns=ns,
                                 experiments=experiments,
                                 hidden_sizes=hidden_sizes, 
                                 optimizer_str=optimizer_str,
                                 simulate_location_list=simulate_location_list,
                                 simulate_scale_list=simulate_scale_list, 
                                 skip_connect=skip_connect) 
    
    csv_names = glob.glob("results/heteroscedastic/location_scale_comparison_*.csv")
    underscore_max_idx_list = [max([pos for pos, char in enumerate(list_item) if char == "_"]) for list_item in csv_names]
    period_max_idx_list  = [max([pos for pos, char in enumerate(list_item) if char == "."]) for list_item in csv_names]
    unique_suffix = max([int(csv_names[idx][(underscore_max_idx_list[idx]+1):period_max_idx_list[idx]]) for idx in range(len(csv_names))]) + 1 # add one to make it unique
    
    filename = "results/heteroscedastic/location_scale_comparison_" + str(unique_suffix) + ".csv"
    df.to_csv(filename) 

if __name__ == "__main__":
    import experiment_heteroscedastic
    import pandas as pd
    import glob

    print("%j")
    log_folder="submitit_log/%j"

    executor = submitit.AutoExecutor(folder=log_folder)

    executor.update_parameters(name="EB", slurm_partition="general", gpus_per_node=1, nodes=8,
                               mem_gb=24, timeout_min=700)
    
    job = executor.submit(location_scale_comparison, m_sim=5, experiments=['c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'], 
                          ns=[100, 200, 400, 800, 1600, 3200, 6400], hidden_sizes=(8,8), simulate_location_list=[False], 
                          simulate_scale_list=[True], skip_connect=False) 

    print(f"job.job_id: {job.job_id}")