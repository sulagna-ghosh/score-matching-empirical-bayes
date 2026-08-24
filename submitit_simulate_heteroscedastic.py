# Executes eight different experiments conducted to get figure 1 in section 6.2 

import submitit
import argparse

def main():
    mse_sure_df = revised_simulate_experiment_location_scale(ns, m_sim_start, m_sim_end, experiment)

    save_path = Path.cwd() / "results" / "heteroscedastic"
    save_path.mkdir(exist_ok=True)

    mse_sure_df.to_csv(str(save_path) + "/experiment_" + experiment + "_" + str(m_sim_start) + "_" + str(m_sim_end) + ".csv")


if __name__ == "__main__":
    from experiment_heteroscedastic import ns
    from experiment_heteroscedastic import revised_simulate_experiment_location_scale
    from pathlib import Path

    print("%j")
    log_folder="submitit_log/%j"

    parser = argparse.ArgumentParser(description='This inputs the experiment (c through h), m_sim_start, and m_sim_end.')
    parser.add_argument("--experiment", type=str, required=True, help="experiment")
    parser.add_argument("--m_sim_start", type=int, required=True, help="m_sim_start")
    parser.add_argument("--m_sim_end", type=int, required=True, help="m_sim_end")
    # Parse the incoming arguments from the command line
    args = parser.parse_args()
    experiment = args.experiment
    print(f"experiment: {experiment}")

    m_sim_start = args.m_sim_start
    m_sim_end = args.m_sim_end

    executor = submitit.AutoExecutor(folder=log_folder)

    executor.update_parameters(name="EB_" + experiment , slurm_partition="general", gpus_per_node=1, nodes=1,
                               mem_gb=24, timeout_min=701)
    
    job = executor.submit(main) 

    print(f"job.job_id: {job.job_id}")