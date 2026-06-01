# Executes eight different experiments conducted to get figure 1 in section 6.2 

import submitit


def save_mse_df(ns=[100, 200, 400, 800, 1600, 3200],
            experiments=["c", "d", "e"],
            optimizer_str="adam", m_sim=1000, B=100):
    
    print("Start location_scale_comparison")

    df = make_df(ns,
            experiments,
            optimizer_str, m_sim, B)
    
    filename = "results/optimization_checks/mse.csv"
    df.to_csv(filename) 

if __name__ == "__main__":
    from optimization_checks import make_df

    print("%j")
    log_folder="submitit_log/%j"

    executor = submitit.AutoExecutor(folder=log_folder)

    executor.update_parameters(name="EB", slurm_partition="general", gpus_per_node=1, nodes=1,
                               mem_gb=24, timeout_min=700)
    
    job = executor.submit(save_mse_df, ns=[100, 200, 400, 800, 1600, 3200],
            experiments=["c", "d", "e"],
            optimizer_str="adam", m_sim=1000, B=100) 

    print(f"job.job_id: {job.job_id}")