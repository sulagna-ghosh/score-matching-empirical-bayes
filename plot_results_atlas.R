library(tidyverse)
# library(arrow) # read feather files
theme_set(theme_bw())
library(latex2exp)
library(RColorBrewer)
dark2_palette <- brewer.pal(8,"Dark2")

NPMLE_color = "#1B9E77"
SURE_PM_color = "#7570B3"
SURE_THING_color = "#E7298A"
SURE_LS_color = "#66A61E"
Z_color = "darkgrey"



# Plot shrinkage rules against sigma #### 

theta_hat = read.csv("results/atlas/shrinkage_rule.csv")

theta_hat_pivot = theta_hat %>%
  select(-X) %>%
  pivot_longer(1:5, names_to = "estimator", values_to = "y") 

theta_hat_pivot = theta_hat_pivot %>%
  mutate(estimator = case_when(estimator == "Z..MLE." ~ "Z (MLE)",
                               estimator == "SURE.PM" ~ "SURE-PM",
                               estimator == "SURE.LS" ~ "SURE-LS",
                               estimator == "SURE.THING" ~ "SURE-THING",
                               TRUE ~ estimator),
         plot_panel = case_when(estimator %in% c("Z (MLE)", "NPMLE", "SURE-PM") ~ 1,
                                estimator %in% c("Z (MLE)", "SURE-THING", "SURE-LS") ~ 2),
         estimator = factor(estimator, levels = c("Z (MLE)", "NPMLE", "SURE-PM", "SURE-LS", "SURE-THING"))) 

library(scales)

shrinkage_plot = theta_hat_pivot %>% 
  filter(plot_panel == 1) %>%
  ggplot(aes(x = (10^(log.sigma))**2, y = y, group = estimator, color = estimator)) +
  geom_point(size = 0.01, alpha=0.9) +
  scale_color_manual(values = c(Z_color, NPMLE_color, SURE_PM_color)) + 
  theme( strip.text.x = element_blank(), legend.position = "bottom", 
         legend.box="vertical",
         text=element_text(size=12), legend.margin=margin(),
        legend.key.size = unit(1,"line"),
        axis.title.x = element_text(vjust=-0.05))   +
  guides(color = guide_legend(override.aes = list(size = 1.5))) +
  ylab(TeX("$\\hat{\\mu}_i$")) + xlab(TeX("$\\sigma_i^2$")) +
  labs(color="") + 
  scale_x_log10(labels = label_log(digits = 2))

shrinkage_plot

## What does Fig 3 in Chen look like for us? ####

# https://jiafengkevinchen.github.io/assets/files/jchen_close.pdf

theta_hat = theta_hat %>%
  mutate(PM_NPMLE_diff = abs(SURE.PM - NPMLE),
         THING_NPMLE_diff = abs(SURE.THING - NPMLE),
         LS_NPMLE_diff = abs(SURE.LS - NPMLE))

theta_hat %>%
  arrange(-LS_NPMLE_diff) %>%
  head(5)

theta_hat %>%
  arrange(-PM_NPMLE_diff) %>%
  head(5)

# Plot prior ####

# Taken from atlas_prior_and_shrinkage.ipynb

pi_hat_npmle = c(0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                 0.0000e+00, 4.9099e-04, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                 0.0000e+00, 0.0000e+00, 8.4995e-04, 1.8166e-09, 0.0000e+00, 0.0000e+00,
                 0.0000e+00, 1.2358e-03, 0.0000e+00, 2.8937e-04, 1.1193e-01, 1.5913e-01,
                 2.1244e-01, 2.0883e-01, 1.4932e-08, 1.2695e-01, 7.7585e-02, 5.7724e-02,
                 3.7040e-10, 2.4492e-10, 2.8456e-02, 1.1694e-02, 0.0000e+00, 0.0000e+00,
                 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                 0.0000e+00, 0.0000e+00, 1.3483e-03, 7.2037e-10, 0.0000e+00, 0.0000e+00,
                 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                 0.0000e+00, 0.0000e+00, 1.0515e-09, 4.8712e-04, 4.1232e-04, 1.5937e-09,
                 5.4153e-10, 2.6135e-10, 3.0940e-10, 7.5589e-10, 2.7498e-09, 1.3693e-04,
                 1.3571e-08, 6.4660e-10, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00)

theta_grid_npmle = c(-0.2884, -0.2797, -0.2710, -0.2622, -0.2535, -0.2448, -0.2361, -0.2274,
                     -0.2187, -0.2100, -0.2013, -0.1926, -0.1839, -0.1752, -0.1665, -0.1577,
                     -0.1490, -0.1403, -0.1316, -0.1229, -0.1142, -0.1055, -0.0968, -0.0881,
                     -0.0794, -0.0707, -0.0619, -0.0532, -0.0445, -0.0358, -0.0271, -0.0184,
                     -0.0097, -0.0010,  0.0077,  0.0164,  0.0251,  0.0338,  0.0426,  0.0513,
                     0.0600,  0.0687,  0.0774,  0.0861,  0.0948,  0.1035,  0.1122,  0.1209,
                     0.1296,  0.1384,  0.1471,  0.1558,  0.1645,  0.1732,  0.1819,  0.1906,
                     0.1993,  0.2080,  0.2167,  0.2254,  0.2341,  0.2429,  0.2516,  0.2603,
                     0.2690,  0.2777,  0.2864,  0.2951,  0.3038,  0.3125,  0.3212,  0.3299,
                     0.3387,  0.3474,  0.3561,  0.3648,  0.3735,  0.3822,  0.3909,  0.3996,
                     0.4083,  0.4170,  0.4257,  0.4344,  0.4432,  0.4519,  0.4606,  0.4693,
                     0.4780,  0.4867,  0.4954,  0.5041,  0.5128,  0.5215,  0.5302,  0.5390,
                     0.5477,  0.5564,  0.5651,  0.5738)

pi_hat_pm = c(7.2010e-04, 4.4188e-05, 5.1695e-04, 6.2180e-04, 4.3891e-04, 2.5237e-04,
              5.3809e-04, 1.5949e-03, 2.4301e-03, 6.1801e-04, 3.2550e-04, 9.9423e-04,
              2.4999e-03, 4.0303e-03, 1.7690e-03, 6.7757e-04, 1.9042e-03, 6.9675e-03,
              5.8486e-03, 1.5445e-03, 1.0270e-03, 2.3892e-03, 5.3731e-03, 1.0844e-02,
              1.4886e-02, 1.1961e-02, 7.2613e-03, 4.3434e-03, 2.9531e-03, 2.7805e-03,
              3.8439e-03, 6.3604e-03, 1.0642e-02, 1.1943e-02, 5.9932e-03, 1.9779e-03,
              6.4287e-04, 1.1015e-03, 3.2671e-03, 1.8918e-02, 1.0614e-01, 2.7040e-02,
              1.0563e-02, 5.6140e-03, 3.4329e-03, 2.2205e-03, 1.4553e-03, 9.3502e-04,
              5.6222e-04, 2.8337e-04, 9.6091e-05, 1.8544e-04, 3.4587e-04, 5.3072e-04,
              7.5311e-04, 1.0279e-03, 1.3732e-03, 1.8135e-03, 2.3833e-03, 3.1329e-03,
              4.1375e-03, 5.5136e-03, 7.4453e-03, 1.0232e-02, 1.4364e-02, 2.0633e-02,
              3.0184e-02, 4.4073e-02, 6.1438e-02, 7.6723e-02, 8.1463e-02, 7.2998e-02,
              5.7479e-02, 4.2210e-02, 3.0368e-02, 2.2008e-02, 1.6239e-02, 1.2220e-02,
              9.3585e-03, 7.2718e-03, 5.7140e-03, 4.5266e-03, 3.6045e-03, 2.8766e-03,
              2.2935e-03, 1.8204e-03, 1.4320e-03, 1.1095e-03, 8.3901e-04, 6.0949e-04,
              4.1199e-04, 2.3819e-04, 8.3810e-05, 6.0100e-05, 1.1122e-03, 1.0474e-02,
              2.6862e-03, 7.2461e-04, 2.2365e-04, 6.5153e-05)

theta_grid_pm = c(-0.3141, -0.3086, -0.1025, -0.0492, -0.0429, -0.0283, -0.0204, -0.0172,
                  -0.0154, -0.0128,  0.0014,  0.0041,  0.0053,  0.0062,  0.0074,  0.0115,
                  0.0154,  0.0164,  0.0171,  0.0185,  0.0260,  0.0284,  0.0294,  0.0301,
                  0.0306,  0.0312,  0.0319,  0.0330,  0.0350,  0.0382,  0.0408,  0.0424,
                  0.0434,  0.0442,  0.0450,  0.0464,  0.0504,  0.0615,  0.0632,  0.0638,
                  0.0640,  0.0642,  0.0644,  0.0646,  0.0648,  0.0652,  0.0657,  0.0664,
                  0.0676,  0.0702,  0.0826,  0.1055,  0.1102,  0.1121,  0.1132,  0.1139,
                  0.1145,  0.1150,  0.1154,  0.1157,  0.1160,  0.1162,  0.1164,  0.1166,
                  0.1167,  0.1169,  0.1170,  0.1171,  0.1173,  0.1174,  0.1175,  0.1176,
                  0.1177,  0.1178,  0.1179,  0.1180,  0.1181,  0.1183,  0.1184,  0.1186,
                  0.1188,  0.1190,  0.1193,  0.1196,  0.1199,  0.1203,  0.1208,  0.1214,
                  0.1222,  0.1233,  0.1251,  0.1289,  0.1431,  0.3276,  0.3551,  0.3604,
                  0.3636,  0.3686,  0.3787,  0.3998)

# pi_hat_pm = c(4.2038e-05, 7.2076e-05, 1.6573e-04, 3.2345e-04, 1.5166e-04, 5.6359e-05,
#               2.0210e-05, 6.6435e-05, 1.7774e-04, 1.6753e-04, 1.4053e-04, 1.9507e-04,
#               2.6089e-04, 2.9923e-04, 3.4886e-04, 4.3542e-04, 5.5036e-04, 6.6524e-04,
#               7.4505e-04, 7.8211e-04, 7.9154e-04, 7.9376e-04, 8.6032e-04, 1.0423e-03,
#               1.2671e-03, 1.4847e-03, 1.7183e-03, 1.9070e-03, 2.0200e-03, 2.2079e-03,
#               2.6337e-03, 3.1373e-03, 3.3164e-03, 3.1520e-03, 3.1325e-03, 3.6482e-03,
#               4.7467e-03, 6.3289e-03, 8.1328e-03, 9.6056e-03, 1.0194e-02, 9.9220e-03,
#               9.3005e-03, 8.7439e-03, 8.3046e-03, 7.7384e-03, 6.7941e-03, 5.7958e-03,
#               6.2386e-03, 1.0190e-02, 2.1387e-02, 4.5698e-02, 4.9681e-02, 2.6404e-02,
#               1.3398e-02, 7.4874e-03, 3.9924e-03, 1.7042e-03, 2.4928e-03, 3.8033e-03,
#               5.4037e-03, 7.4331e-03, 1.0052e-02, 1.3460e-02, 1.7888e-02, 2.3557e-02,
#               3.0559e-02, 3.8619e-02, 4.6782e-02, 5.3322e-02, 5.6370e-02, 5.5103e-02,
#               5.0323e-02, 4.3720e-02, 3.6814e-02, 3.0489e-02, 2.5079e-02, 2.0609e-02,
#               1.6971e-02, 1.4023e-02, 1.1630e-02, 9.6771e-03, 8.0707e-03, 6.7379e-03,
#               5.6212e-03, 4.6758e-03, 3.8666e-03, 3.1657e-03, 2.5506e-03, 2.0027e-03,
#               1.5059e-03, 1.0437e-03, 5.9184e-04, 2.2026e-04, 8.3487e-04, 1.5209e-03,
#               2.1470e-03, 2.7801e-03, 3.5039e-03, 4.4119e-03)
# 
# theta_grid_pm = c(-3.6618e-01, -3.4025e-01, -3.2663e-01, -3.1930e-01, -3.1277e-01,
#                   -3.0241e-01, -2.7552e-01, -1.1519e-01, -1.0412e-01, -9.3036e-02,
#                   -7.3994e-02, -5.8163e-02, -4.9047e-02, -4.1481e-02, -3.4655e-02,
#                   -2.9089e-02, -2.4693e-02, -2.1024e-02, -1.7690e-02, -1.4396e-02,
#                   -1.0954e-02, -7.2471e-03, -3.4695e-03, -3.1349e-04,  2.1545e-03,
#                   4.2874e-03,  6.2278e-03,  8.0885e-03,  1.0018e-02,  1.2056e-02,
#                   1.3999e-02,  1.5757e-02,  1.7524e-02,  1.9601e-02,  2.2173e-02,
#                   2.4796e-02,  2.6938e-02,  2.8625e-02,  3.0045e-02,  3.1352e-02,
#                   3.2662e-02,  3.4073e-02,  3.5662e-02,  3.7483e-02,  3.9575e-02,
#                   4.2018e-02,  4.5087e-02,  4.9484e-02,  5.5504e-02,  5.9784e-02,
#                   6.1988e-02,  6.3228e-02,  6.4075e-02,  6.4850e-02,  6.5739e-02,
#                   6.6951e-02,  6.9162e-02,  7.8538e-02,  1.0625e-01,  1.0991e-01,
#                   1.1163e-01,  1.1273e-01,  1.1354e-01,  1.1418e-01,  1.1472e-01,
#                   1.1517e-01,  1.1558e-01,  1.1595e-01,  1.1628e-01,  1.1660e-01,
#                   1.1691e-01,  1.1721e-01,  1.1751e-01,  1.1782e-01,  1.1814e-01,
#                   1.1847e-01,  1.1883e-01,  1.1921e-01,  1.1962e-01,  1.2006e-01,
#                   1.2056e-01,  1.2110e-01,  1.2172e-01,  1.2241e-01,  1.2322e-01,
#                   1.2416e-01,  1.2528e-01,  1.2667e-01,  1.2844e-01,  1.3082e-01,
#                   1.3429e-01,  1.4018e-01,  1.5437e-01,  2.7540e-01,  3.3852e-01,
#                   3.5239e-01,  3.5863e-01,  3.6224e-01,  3.6458e-01,  3.6618e-01)

prior_data = data.frame("prob" = c(pi_hat_npmle, pi_hat_pm),
           "grid" = c(theta_grid_npmle, theta_grid_pm),
           "model" = c(rep("NPMLE", 100), rep("SURE-PM", 100)),
           "idx" = c(rep(0, 100), 1:100))

prior_plot = prior_data %>%
  filter(prob > 0) %>%
  ggplot(aes(x = grid, y = prob, group = model)) +
  geom_point(aes(color = model), size=1, alpha=0.9) +
  geom_segment(aes(color=model, xend=grid, yend= 0), alpha=0.9, size = 0.5) + 
  scale_color_manual(values = c(NPMLE_color, SURE_PM_color)) +
  theme(legend.position="bottom",
        text=element_text(size=12), legend.margin=margin(),
        legend.key.size = unit(1,"line")) +
  xlab(TeX("$u_j\n\n\n$")) +
  ylab(TeX("$pi_j")) 



# Figure 4 #### 


library(ggpubr)
g = ggarrange(shrinkage_plot, prior_plot, ncol=2, common.legend = TRUE, legend="bottom")
ggsave("results/figures/figure_4.png", g, width=6, height=3.5)


# Table of data fission results (Table 3) ####

mean_squared_standard_error = 0.0029098331424100635
# Taken from atlas_prior_and_shrinkage.ipynb

B = 25 # number of replicates

# Get idx
fission_idx = read.csv("results/atlas/data_fission_mse.csv") %>%
  filter(mosek_fail=="False", !is.na(THING_bivariate)) %>%
  head(B) %>% pull(X)

# Get CLOSE MSE
# (CLOSE only runs locally so it's a separate csv)
improvement_over_MLE_close = read.csv("results/atlas/sse_close.csv") %>% 
  filter(X %in% fission_idx) %>%
  mutate(CLOSE = CLOSE / 10056,
         CLOSE.residualized = CLOSE.residualized / 10056) %>% select(-X)

# from submitit_data_fission.py
improvement_over_MLE = read.csv("results/atlas/data_fission_mse.csv") %>%
  filter(mosek_fail=="False", !is.na(THING_bivariate)) %>%
  head(B) %>%
  select(-mosek_fail, -X) %>%
  bind_cols(improvement_over_MLE_close) %>%
  mutate(rel_PM = (MLE - PM) / (MLE - NPMLE),
         rel_NPMLE = (MLE - NPMLE) / (MLE - NPMLE),
         rel_THING_sigma = (MLE - THING_sigma) / (MLE - NPMLE),
         rel_LS_sigma = (MLE - LS_sigma) / (MLE - NPMLE),
         rel_THING_bivariate = (MLE - THING_bivariate) / (MLE - NPMLE),
         rel_LS_bivariate = (MLE - LS_bivariate) / (MLE - NPMLE),
         rel_THING_full = (MLE - THING_full) / (MLE - NPMLE),
         rel_LS_full = (MLE - LS_full) / (MLE - NPMLE),
         rel_CLOSE_sigma = (MLE - CLOSE) / (MLE - NPMLE),
         rel_CLOSE_full = (MLE - CLOSE.residualized) / (MLE - NPMLE)) %>%
  select(starts_with("rel_")) %>%
  pivot_longer(starts_with("rel_"), names_to = "model", values_to = "relative_improvement") 

improvement_over_MLE %>%
  group_by(model) %>%
  summarize(mean_improvement = mean(relative_improvement),
            se_improvement = sd(relative_improvement)/sqrt(B)) %>%
  filter(str_detect(model, "_bivariate", negate=TRUE)) %>% view()

# Scratch ####

# if |u_j - u_j'| < 0.001, 
# then merge them and assign 
# the merged thing a probability \pi_j + \pi_j'

merged_pi_hat_npmle = c()
merged_theta_grid_npmle = c()
merged_pi_hat_pm = c()
merged_theta_grid_pm = c()

merge_epsilon = 0.01

for (j in seq(1, 100, 2)){ # merge odd idx
  
  if (abs(theta_grid_npmle[j] - theta_grid_npmle[j+1]) < merge_epsilon){
    
    merged_theta_grid_npmle = c(merged_theta_grid_npmle, theta_grid_npmle[j] + theta_grid_npmle[j+1])
    merged_pi_hat_npmle = c(merged_pi_hat_npmle, pi_hat_npmle[j] + pi_hat_npmle[j+1])
    
  } else {
    
    merged_theta_grid_npmle = c(merged_theta_grid_npmle, theta_grid_npmle[j], theta_grid_npmle[j+1])
    merged_pi_hat_npmle = c(merged_pi_hat_npmle, pi_hat_npmle[j], pi_hat_npmle[j+1])
    
  }
  
  if (abs(theta_grid_pm[j] - theta_grid_pm[j+1]) < merge_epsilon){
    
    merged_theta_grid_pm = c(merged_theta_grid_pm, theta_grid_pm[j] + theta_grid_pm[j+1])
    merged_pi_hat_pm = c(merged_pi_hat_pm, pi_hat_pm[j] + pi_hat_pm[j+1])
    
  } else {
    
    merged_theta_grid_pm = c(merged_theta_grid_pm, theta_grid_pm[j], theta_grid_pm[j+1])
    merged_pi_hat_pm = c(merged_pi_hat_pm, pi_hat_pm[j], pi_hat_pm[j+1])
    
  }
  
}

merged_prior_data = data.frame("prob" = c(merged_pi_hat_npmle, merged_pi_hat_pm),
                               "grid" = c(merged_theta_grid_npmle, merged_theta_grid_pm),
                               "model" = c(rep("NPMLE", length(merged_pi_hat_npmle)), rep("SURE-PM", length(merged_pi_hat_pm))))

prior_plot = ggplot(merged_prior_data, aes(x = grid, y = prob, group = model)) +
  geom_point(aes(color = model), size=1, alpha=0.9) +
  geom_segment(aes(color=model, xend=grid, yend= 0), alpha=0.9, size = 0.5) + 
  scale_color_manual(values = c(NPMLE_color, SURE_PM_color)) +
  theme(legend.position="bottom",
        text=element_text(size=12), legend.margin=margin(),
        legend.key.size = unit(3,"line")) +
  xlab(TeX("$u_j\n\n\n$")) +
  ylab(TeX("$pi_j")) 