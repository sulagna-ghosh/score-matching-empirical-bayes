# Executes experiments to comapre different optimizers and activation functions 

import experiments_xie
import pandas as pd
import torch as tr

# misspec: ADAM vs BFGS
misspec_result = experiments_xie.bfgs_versus_adam(n=100000, experiments = ["d", "d5"], model="misspec", m_sim=1)
# print(misspec_result)

misspec_result.to_csv("results/adam_vs_bfgs_misspec_100k.csv")

# wellspec: ADAM (ReLu) vs BFGS (ELU) vs BFGS (SiLU/swish)
wellspec_result = experiments_xie.bfgs_versus_adam(n=100000, experiments = ["d", "d5"], model="wellspec", m_sim=2)
print(wellspec_result)

wellspec_result.to_csv("results/adam_vs_bfgs_wellspec_100k.csv")