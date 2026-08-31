import submitit

if __name__ == "__main__":
    import torch as tr
    import numpy as np
    import pandas as pd
    from simulate_data import device
    from experiment_heteroscedastic import read_experiment_dict, read_slice_of_experiment, master_seed, experiments
    from models import model_theta_pi_sure
    from train import train_and_evaluate_npmle, train_no_covariates

    # TODO: write function to see the weights and grid

    # TODO: be able to pick specific values of m_sim

    def train_estimators_and_get_sure(slice_of_experiment, use_location, use_scale, n, B,
                                    m_sim, append_metrics_to_lists):

        theta, Z, X = slice_of_experiment
        Z = Z.to(device)
        X = X.to(device)

        ####### Estimators #######

        ### NPMLE ###
        output_npmle = train_and_evaluate_npmle(n, B, Z, theta, X)
        mse_npmle, sure_npmle, found_npmle_solution, pi_hat_npmle = output_npmle
        append_metrics_to_lists(sure_npmle, "NPMLE",  m_sim, n)

        ### SURE-PM ###

        # Standard (uniform weights & grid initialization)
        pm_output = train_no_covariates(n, B, Z, theta, X, opt_objective = 'both', 
                                init_val_theta = tr.log(tr.Tensor([1.5])), init_val_pi = tr.log(tr.Tensor([1.5])),
                                use_location=use_location, use_scale=use_scale, randomly_initialize_theta_pi=False,
                                device=device, optimizer_str="adam",
                                lr=1e-2, n_iter=4000)
        model, sures, _, _, _ = pm_output
        append_metrics_to_lists(sures, "SURE-PM uniform",  m_sim, n)


        # NPMLE-initialization
        if found_npmle_solution:
            # Grid of NPMLE: B evenly spaced points between smallest and largest Z value.
            pm_npmleinit_output = train_no_covariates(n, B, Z, theta, X, opt_objective = 'both', 
                                    init_val_theta = tr.log(tr.Tensor([1.5])), init_val_pi = tr.log(tr.Tensor(pi_hat_npmle)),
                                    use_location=use_location, use_scale=use_scale, randomly_initialize_theta_pi=False,
                                    device=device, optimizer_str="adam",
                                    lr=1e-2, n_iter=4000, initialize_at_npmle=True)
            model, sures, _, _, _ = pm_npmleinit_output
            append_metrics_to_lists(sures, "SURE-PM NPMLEinit",  m_sim, n)
        else:
            append_metrics_to_lists(np.nan, "SURE-PM NPMLEinit",  m_sim, n)

        # 10 iterations of random SURE initialization
        tr.manual_seed(master_seed)
        for _ in range(10):
            # randomly_initialize_theta_pi=True
            pm_random_output = train_no_covariates(n, B, Z, theta, X, opt_objective = 'both', 
                                init_val_theta = tr.log(tr.Tensor([1.5])), init_val_pi = tr.log(tr.Tensor([1.5])),
                                use_location=use_location, use_scale=use_scale, randomly_initialize_theta_pi=True,
                                device=device, optimizer_str="adam",
                                lr=1e-2, n_iter=4000)
            model, sures, _, _, _ = pm_random_output
            append_metrics_to_lists(sures, "SURE-PM random",  m_sim, n)

    def save_csv(experiment_str, n=1000, total_m_sim=10,
                use_scale=True, use_location=False, B=100):

        print(f"n: {n}")

        # Load data
        experiment_dict = read_experiment_dict(experiment_str)

        estimator_list, sure_list = [], []
        sure_iter_1st_quartile, sure_iter_2nd_quartile, sure_iter_3rd_quartile = [], [], []
        m_sim_list, n_list = [], []

        def append_metrics_to_lists(sures, estimator,  m_sim, n):
            assert estimator in ["NPMLE", "SURE-PM uniform", "SURE-PM NPMLEinit", "SURE-PM random"]
            estimator_list.append(estimator)

            if isinstance(sures, float):
                sure_list.append(sures)
                sure_iter_1st_quartile.append(np.nan)
                sure_iter_2nd_quartile.append(np.nan)
                sure_iter_3rd_quartile.append(np.nan)
            else:
                sure_list.append(sures[-1])
                sure_iter_1st_quartile.append(sures[1000])
                sure_iter_2nd_quartile.append(sures[2000])
                sure_iter_3rd_quartile.append(sures[3000])

            m_sim_list.append(m_sim)
            n_list.append(n)

        # Read the first <total_m_sim> slices of data for that experiment parameter
        for m_sim in range(total_m_sim):
            print(f"m_sim: {m_sim}")
            slice_of_experiment = read_slice_of_experiment(experiment_dict, n=n, m=m_sim)
            train_estimators_and_get_sure(slice_of_experiment, use_scale, use_location, n, B,
                                        m_sim, append_metrics_to_lists)

        df = pd.DataFrame({'estimator': estimator_list,
                            'sure': sure_list,
                            'sure_1000th_iter': sure_iter_1st_quartile,
                            'sure_2000th_iter': sure_iter_2nd_quartile,
                            'sure_3000th_iter': sure_iter_3rd_quartile})

        print(f"df: {df}")
        filename = experiment_str + "_" + str(n) + "_different_inits.csv"
        df.to_csv("results/optimization_checks/" + filename)

    def main():
        for experiment_str in experiments:
            save_csv(experiment_str=experiment_str, n=1000, total_m_sim=10)

    log_folder="submitit_log/%j"
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(name="EB", slurm_partition="general", gpus_per_node=1, nodes=1,
                               mem_gb=3, timeout_min=600)

    job = executor.submit(main) 

    print(f"job.job_id: {job.job_id}")








####### Anticipated questions #######

# How are we setting the weights and grid?
## get_theta_grid_and_pi
## use_location and use_scale hyperparameter