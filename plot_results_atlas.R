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

theta_hat = read.csv("results/one_run_heteroscedastic/shrinkage_rule.csv") %>% select(-X)

theta_hat_pivot = theta_hat %>%
  pivot_longer(1:5, names_to = "estimator", values_to = "y") 

theta_hat_pivot = theta_hat_pivot %>%
  mutate(estimator = case_when(estimator == "Z..MLE." ~ "Z (MLE)",
                               estimator == "SURE.PM" ~ "SURE-PM",
                               estimator == "SURE.LS" ~ "SURE-LS",
                               estimator == "SURE.THING" ~ "SURE-THING",
                               TRUE ~ estimator),
         plot_panel = case_when(estimator %in% c("Z (MLE)", "NPMLE", "SURE-PM") ~ 1,
                                estimator %in% c("Z (MLE)", "SURE-THING", "SURE-LS") ~ 2)) 

shrinkage_plot = theta_hat_pivot %>% 
  filter(plot_panel == 1) %>%
  ggplot(aes(x = log.sigma, y = y, group = estimator, color = estimator)) +
  geom_point(size = 0.01, alpha=0.9) +
  scale_color_manual(values = c(Z_color, NPMLE_color, SURE_PM_color, 
                                SURE_THING_color, SURE_LS_color)) + 
  theme( strip.text.x = element_blank(), legend.position = "bottom", 
         legend.box="vertical",
         text=element_text(size=12), legend.margin=margin(),
        legend.key.size = unit(3,"line"))   +
  ylab(TeX("$\\hat{\\mu}_i$")) + xlab(TeX("$log(\\sigma_i)$")) +
  labs(color="") +
  guides(color = guide_legend(override.aes = list(size=1))) 


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

theta_grid_npmle = c(0.5738,  0.5651,  0.5564,  0.5477,  0.5390,  0.5302,  0.5215,  0.5128,
                     0.5041,  0.4954,  0.4867,  0.4780,  0.4693,  0.4606,  0.4519,  0.4432,
                     0.4344,  0.4257,  0.4170,  0.4083,  0.3996,  0.3909,  0.3822,  0.3735,
                     0.3648,  0.3561,  0.3474,  0.3387,  0.3299,  0.3212,  0.3125,  0.3038,
                     0.2951,  0.2864,  0.2777,  0.2690,  0.2603,  0.2516,  0.2429,  0.2341,
                     0.2254,  0.2167,  0.2080,  0.1993,  0.1906,  0.1819,  0.1732,  0.1645,
                     0.1558,  0.1471,  0.1384,  0.1296,  0.1209,  0.1122,  0.1035,  0.0948,
                     0.0861,  0.0774,  0.0687,  0.0600,  0.0513,  0.0426,  0.0338,  0.0251,
                     0.0164,  0.0077, -0.0010, -0.0097, -0.0184, -0.0271, -0.0358, -0.0445,
                     -0.0532, -0.0619, -0.0707, -0.0794, -0.0881, -0.0968, -0.1055, -0.1142,
                     -0.1229, -0.1316, -0.1403, -0.1490, -0.1577, -0.1665, -0.1752, -0.1839,
                     -0.1926, -0.2013, -0.2100, -0.2187, -0.2274, -0.2361, -0.2448, -0.2535,
                     -0.2622, -0.2710, -0.2797, -0.2884)

pi_hat_pm = c(4.2038e-05, 7.2076e-05, 1.6573e-04, 3.2345e-04, 1.5166e-04, 5.6359e-05,
              2.0210e-05, 6.6435e-05, 1.7774e-04, 1.6753e-04, 1.4053e-04, 1.9507e-04,
              2.6089e-04, 2.9923e-04, 3.4886e-04, 4.3542e-04, 5.5036e-04, 6.6524e-04,
              7.4505e-04, 7.8211e-04, 7.9154e-04, 7.9376e-04, 8.6032e-04, 1.0423e-03,
              1.2671e-03, 1.4847e-03, 1.7183e-03, 1.9070e-03, 2.0200e-03, 2.2079e-03,
              2.6337e-03, 3.1373e-03, 3.3164e-03, 3.1520e-03, 3.1325e-03, 3.6482e-03,
              4.7467e-03, 6.3289e-03, 8.1328e-03, 9.6056e-03, 1.0194e-02, 9.9220e-03,
              9.3005e-03, 8.7439e-03, 8.3046e-03, 7.7384e-03, 6.7941e-03, 5.7958e-03,
              6.2386e-03, 1.0190e-02, 2.1387e-02, 4.5698e-02, 4.9681e-02, 2.6404e-02,
              1.3398e-02, 7.4874e-03, 3.9924e-03, 1.7042e-03, 2.4928e-03, 3.8033e-03,
              5.4037e-03, 7.4331e-03, 1.0052e-02, 1.3460e-02, 1.7888e-02, 2.3557e-02,
              3.0559e-02, 3.8619e-02, 4.6782e-02, 5.3322e-02, 5.6370e-02, 5.5103e-02,
              5.0323e-02, 4.3720e-02, 3.6814e-02, 3.0489e-02, 2.5079e-02, 2.0609e-02,
              1.6971e-02, 1.4023e-02, 1.1630e-02, 9.6771e-03, 8.0707e-03, 6.7379e-03,
              5.6212e-03, 4.6758e-03, 3.8666e-03, 3.1657e-03, 2.5506e-03, 2.0027e-03,
              1.5059e-03, 1.0437e-03, 5.9184e-04, 2.2026e-04, 8.3487e-04, 1.5209e-03,
              2.1470e-03, 2.7801e-03, 3.5039e-03, 4.4119e-03)

theta_grid_pm = c(-3.6618e-01, -3.4025e-01, -3.2663e-01, -3.1930e-01, -3.1277e-01,
                  -3.0241e-01, -2.7552e-01, -1.1519e-01, -1.0412e-01, -9.3036e-02,
                  -7.3994e-02, -5.8163e-02, -4.9047e-02, -4.1481e-02, -3.4655e-02,
                  -2.9089e-02, -2.4693e-02, -2.1024e-02, -1.7690e-02, -1.4396e-02,
                  -1.0954e-02, -7.2471e-03, -3.4695e-03, -3.1349e-04,  2.1545e-03,
                  4.2874e-03,  6.2278e-03,  8.0885e-03,  1.0018e-02,  1.2056e-02,
                  1.3999e-02,  1.5757e-02,  1.7524e-02,  1.9601e-02,  2.2173e-02,
                  2.4796e-02,  2.6938e-02,  2.8625e-02,  3.0045e-02,  3.1352e-02,
                  3.2662e-02,  3.4073e-02,  3.5662e-02,  3.7483e-02,  3.9575e-02,
                  4.2018e-02,  4.5087e-02,  4.9484e-02,  5.5504e-02,  5.9784e-02,
                  6.1988e-02,  6.3228e-02,  6.4075e-02,  6.4850e-02,  6.5739e-02,
                  6.6951e-02,  6.9162e-02,  7.8538e-02,  1.0625e-01,  1.0991e-01,
                  1.1163e-01,  1.1273e-01,  1.1354e-01,  1.1418e-01,  1.1472e-01,
                  1.1517e-01,  1.1558e-01,  1.1595e-01,  1.1628e-01,  1.1660e-01,
                  1.1691e-01,  1.1721e-01,  1.1751e-01,  1.1782e-01,  1.1814e-01,
                  1.1847e-01,  1.1883e-01,  1.1921e-01,  1.1962e-01,  1.2006e-01,
                  1.2056e-01,  1.2110e-01,  1.2172e-01,  1.2241e-01,  1.2322e-01,
                  1.2416e-01,  1.2528e-01,  1.2667e-01,  1.2844e-01,  1.3082e-01,
                  1.3429e-01,  1.4018e-01,  1.5437e-01,  2.7540e-01,  3.3852e-01,
                  3.5239e-01,  3.5863e-01,  3.6224e-01,  3.6458e-01,  3.6618e-01)

prior_data = data.frame("prob" = c(pi_hat_npmle, pi_hat_pm),
           "grid" = c(theta_grid_npmle, theta_grid_pm),
           "model" = c(rep("NPMLE", 100), rep("SURE-PM", 100)))

prior_plot = ggplot(prior_data, aes(x = grid, y = prob, group = model)) +
  geom_point(aes(color = model), size=1, alpha=0.9) +
  geom_segment(aes(color=model, xend=grid, yend= 0), alpha=0.9, size = 0.5) + 
  scale_color_manual(values = c(NPMLE_color, SURE_PM_color)) +
  theme(legend.position="bottom",
        text=element_text(size=12), legend.margin=margin(),
        legend.key.size = unit(3,"line")) +
  xlab(TeX("$mu_j\n\n\n$")) +
  ylab(TeX("$pi_j"))

# Figure 4 #### 


library(ggpubr)
g = ggarrange(shrinkage_plot, prior_plot, ncol=2, common.legend = TRUE, legend="bottom")
ggsave("results/figures/figure_4.png", g, width=6, height=3.5)


# Table of data fission results (Table 3) ####

mean_squared_standard_error = 0.0029098331424100635
# Taken from atlas_prior_and_shrinkage.ipynb

B = 25 # number of replicates

improvement_over_MLE = read.csv("results/atlas/data_fission_mse.csv") %>%
  filter(mosek_fail=="False") %>%
  head(B) %>%
  select(-mosek_fail, -X) %>%
  mutate(rel_PM = (MLE - PM) / (MLE - 2*mean_squared_standard_error),
         rel_NPMLE = (MLE - NPMLE) / (MLE - 2*mean_squared_standard_error)) %>%
  select(4:5) %>%
  pivot_longer(1:2, names_to = "model", values_to = "relative_improvement") 

improvement_over_MLE %>%
  group_by(model) %>%
  summarize(mean_improvement = mean(relative_improvement),
            se_improvement = sd(relative_improvement)/sqrt(B)) 
