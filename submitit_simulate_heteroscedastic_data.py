# Executes eight different experiments conducted to get figure 1 in section 6.2 

import submitit


if __name__ == "__main__":
    from experiment_heteroscedastic import master_seed, m_sim, ns, experiments
    from experiment_heteroscedastic import generate_and_save_synthetic_data

    print("%j")
    log_folder="submitit_log/%j"

    executor = submitit.AutoExecutor(folder=log_folder)

    executor.update_parameters(name="EB", slurm_partition="general", gpus_per_node=0, nodes=1,
                               mem_gb=24, timeout_min=700)
    
    job = executor.submit(generate_and_save_synthetic_data, master_seed, ns, m_sim, experiments) 

    print(f"job.job_id: {job.job_id}")