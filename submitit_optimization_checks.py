import submitit


def save_mse_df(start_seed, end_seed,
        ns=[100, 200, 400],
            experiments=["c", "d", "e"],
            optimizer_str="adam", B=100):
    
    print("Start location_scale_comparison")

    df = make_df(start_seed, end_seed, ns,
            experiments,
            optimizer_str,  B)
    
    filename = "results/optimization_checks/mse_" + str(start_seed) + "_" + str(end_seed) + ".csv"
    df.to_csv(filename) 

if __name__ == "__main__":
    from optimization_checks import make_df

    print("%j")
    log_folder="submitit_log/%j"

    executor = submitit.AutoExecutor(folder=log_folder)

    executor.update_parameters(name="EB", slurm_partition="general", gpus_per_node=2, nodes=1,
                               mem_gb=24, timeout_min=700)
    
    job = executor.submit(save_mse_df, start_seed = 3000, end_seed = 3199, ns=[100, 400],
            experiments=["c", "d", "e"],
            optimizer_str="adam", B=100) 

    print(f"job.job_id: {job.job_id}")