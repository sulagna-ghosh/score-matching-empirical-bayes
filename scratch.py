from experiment_homoscedastic_copy import n, homoskedastic_sigma
from experiment_homoscedastic_copy import problems_binary_theta, make_discrete_homoscedastic_df
from experiment_homoscedastic_copy import problems_normal_theta, make_normal_homoscedastic_df


normal_df = make_normal_homoscedastic_df(problems_normal_theta, homoskedastic_sigma, m_sim=1)
print(f"normal_df: {normal_df}\n")

discrete_df = make_discrete_homoscedastic_df(problems_binary_theta, homoskedastic_sigma, m_sim=1)
print(f"discrete_df: {discrete_df}\n")


# from models import model_theta_pi_sure
# from experiment_heteroscedastic import read_experiment_dict, read_slice_of_experiment
# from train import train_and_evaluate_npmle
# import numpy as np
# from scipy.special import softmax
# import torch as tr

# print("Read experiment data\n")
# experiment_dict_c = read_experiment_dict('c')
# n = 100
# m = 0
# theta, Z, X = read_slice_of_experiment(experiment_dict_c, n, m)
# B = 100

# # Initialize SURE-PM model, for some arbitrary init_val_pi, init_val_theta 
# # that will be set afterwards
# sure_pm_model = model_theta_pi_sure(Z, B, use_location=False, use_scale=True,
#                                     init_val_theta = tr.log(tr.Tensor([1.5])), init_val_pi = tr.log(tr.Tensor([1.5])),
#                                     initialize_at_npmle=True,
#                                     quantile_IQR=1)
# # theta_grid, pi_grid = sure_pm_model.get_theta_grid_and_pi(n, B)

# # NPMLE grid
# theta_diff = (1/(B-1))*np.ones(B-1) 
# theta_cum = np.cumsum(theta_diff, axis = 0)*(max(Z.detach().cpu().numpy())-min(Z.detach().cpu().numpy())) 
# theta_cum = np.concatenate((np.zeros(1), theta_cum))
# theta_grid_npmle = theta_cum + min(Z.detach().cpu().numpy())
# print(f"NPMLE theta_grid: {theta_grid_npmle}") # Correct

# print("Find NPMLE solution")
# MSE_NPMLE, SURE_NPMLE, found_NPMLE_solution, pi_hat_NPMLE = train_and_evaluate_npmle(n, B, Z, theta, X)

# # Convert the NPMLE locations and weights to NumPy.
# u_npmle = theta_grid_npmle # already numpy
# p = pi_hat_NPMLE.detach().cpu().double().numpy()
# p = np.maximum(p, 0)
# p /= p.sum()

# K = sure_pm_model.B
# m = sure_pm_model.location.detach().item() # Keep this value unchanged.
# assert not sure_pm_model.location.requires_grad
# assert sure_pm_model.log_scale.requires_grad
# assert K >= 3

# # Retain the largest positive weights.
# keep = np.flatnonzero(p > 0)
# keep = keep[np.argsort(p[keep])[-(K - 2):]]
# discarded_mass = float(1 - p[keep].sum())
# retained_weights = p[keep] / p[keep].sum()

# # Symmetric boundaries around the EXISTING fixed location.
# radius = float(np.max(np.abs(u_npmle - m)))
# assert radius > 0
# u = np.unique(np.r_[m - radius, u_npmle[keep], m + radius])
# weights = np.zeros(len(u))
# weights[np.searchsorted(u, u_npmle[keep])] = retained_weights

# # Fill unused slots with zero-weight particles in the largest gaps.
# while len(u) < K:
#     j = np.argmax(np.diff(u))
#     midpoint = u[j] + (u[j + 1] - u[j]) / 2
#     u = np.insert(u, j + 1, midpoint)
#     weights = np.insert(weights, j + 1, 0.0)

# # Initialize the actual spacings, scale, and smoothed weights.
# with tr.no_grad():
#     sure_pm_model.theta_real_vals.copy_(
#         sure_pm_model.theta_real_vals.new_tensor(np.log(np.diff(u)))
#         )
#     sure_pm_model.log_scale.fill_(np.log(2 * radius))
#     sure_pm_model.pi_real_vals.copy_(
#         sure_pm_model.pi_real_vals.new_tensor(
#             np.log(0.99 * weights + 0.01 / K)
#             )
#             )

# print(f"NPMLE mass discarded: {discarded_mass:.3g}")

# print(f"keep: {keep}")
# keep = np.flatnonzero(p > 0)
# print(f"keep after changing: {keep}")

# print("PRINT GRID OF SURE PM")
# theta_grid, pi_grid = sure_pm_model.get_theta_grid_and_pi(n, B)
# print(f"theta_grid: {theta_grid[0]}")

# print(f"NPMLE theta_grid: {theta_grid_npmle}") # Correct

# print("GRID POINTS WITH NON ZERO PROBABILITY MASS")
# print(f"NPMLE: {theta_grid_npmle[keep]}")
# print(f"SURE-PM: {theta_grid[0][keep]}")

# print(sure_pm_model.max_Z)

# # # NPMLE solution to initialize at
# # fix_pi_hat_NPMLE = 0.99*pi_hat_NPMLE + 0.01*1/B
# # # print(f"fix_pi_hat_NPMLE: {fix_pi_hat_NPMLE}") # Correct
# # log_of_fix_pi_hat_npmle = np.log(fix_pi_hat_NPMLE)
# # softmax_log_of_pi_hat_NPMLE = softmax(log_of_fix_pi_hat_npmle)
# # # print(f"softmax_log_of_pi_hat_NPMLE: {softmax_log_of_pi_hat_NPMLE}") # Correct



# # print(f"theta_grid: {theta_grid[0].detach().cpu().numpy()}")
# # # print(f"pi_grid") # Correct
# # # for i in range(len(pi_grid[0])):
# # #     print(pi_grid[0,i]) # Correct
# # print(f"sure_pm_model.location: {sure_pm_model.location}")

# # print(np.isclose(theta_grid_npmle, theta_grid[0].detach().cpu().numpy()))
# # # Returns TRUE for all grid points

# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(2, 1)
# axs[0].stem(theta_grid_npmle[keep], retained_weights, label="NPMLE")
# axs[1].stem(theta_grid[0][keep].detach().numpy(), pi_grid[0][keep].detach().numpy(), label="PM")

# plt.show()
# plt.savefig('tmp.png')