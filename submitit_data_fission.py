import submitit


def simulate_sse_without_close():
    
    df = data_fission.fission_mse_without_close()
    df.to_csv('results/atlas/data_fission_mse.csv')

if __name__ == "__main__":
    import atlas_data_fission
    import pandas as pd

    print("%j")
    log_folder="../submitit_log/%j"

    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(name="EB", slurm_partition="general", gpus_per_node=1, nodes=1,
                               mem_gb=15, timeout_min=700)

    job = executor.submit(simulate_sse_without_close)

    print(f"job.job_id: {job.job_id}")

