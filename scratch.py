from models import model_theta_pi_sure
from experiment_heteroscedastic import read_experiment_dict, read_slice_of_experiment
from train import train_and_evaluate_npmle
import numpy as np
from scipy.special import softmax

print("Read experiment data\n")
experiment_dict_c = read_experiment_dict('c')
n = 100
m = 0
theta, Z, X = read_slice_of_experiment(experiment_dict_c, n, m)
B = 100

print("Find NPMLE solution")
MSE_NPMLE, SURE_NPMLE, found_NPMLE_solution, pi_hat_NPMLE = train_and_evaluate_npmle(n, B, Z, theta, X)

# NPMLE solution to initialize at
fix_pi_hat_NPMLE = 0.99*pi_hat_NPMLE + 0.01*1/B
# print(f"fix_pi_hat_NPMLE: {fix_pi_hat_NPMLE}") # Correct
log_of_fix_pi_hat_npmle = np.log(fix_pi_hat_NPMLE)
softmax_log_of_pi_hat_NPMLE = softmax(log_of_fix_pi_hat_npmle)
# print(f"softmax_log_of_pi_hat_NPMLE: {softmax_log_of_pi_hat_NPMLE}") # Correct

# Grid
theta_diff = (1/(B-1))*np.ones(B-1) 
theta_cum = np.cumsum(theta_diff, axis = 0)*(max(Z.detach().cpu().numpy())-min(Z.detach().cpu().numpy())) 
theta_cum = np.concatenate((np.zeros(1), theta_cum))
theta_grid = theta_cum + min(Z.detach().cpu().numpy())
# print(f"theta_grid: {theta_grid}") # Correct
# print(f"len(theta_grid): {len(theta_grid)}")
# print(f"min(Z): {min(Z)}")
# print(f"max(Z): {max(Z)}")

print("Initialize SURE-PM model")
sure_pm_model = model_theta_pi_sure(Z, B, 
                                    init_val_pi=log_of_fix_pi_hat_npmle, init_val_theta=1,
                                    use_location=False, use_scale=True,
                                    initialize_at_npmle=True)
theta_grid, pi_grid = sure_pm_model.get_theta_grid_and_pi(n, B)

print(f"theta_grid: {theta_grid}")
# print(f"pi_grid") # Correct
# for i in range(len(pi_grid[0])):
#     print(pi_grid[0,i]) # Correct

