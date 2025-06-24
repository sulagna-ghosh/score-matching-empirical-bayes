setwd("~/Documents/Github/empirical-bayes/results")
library(tidyverse)
library(ggh4x) # facetted_pos_scales
library(ggpubr) # arrange multiple figures together
library(latex2exp)
library(RColorBrewer)
dark2_palette <- brewer.pal(8,"Dark2")
theme_set(theme_bw())

# Read path ####

seed_str = "37455999"
read_path = paste("xie_checks_", seed_str, "/", sep="")

# Marginals ###
debug_marginals_df = read.csv(paste(read_path,"location_scale_marginals.csv", sep="")) %>% 
  select(-X) 

# 6400 n x 4 location scale x 5 experments x 3 variances
debug_marginals_df$use_scale = as.logical(debug_marginals_df$use_scale)
debug_marginals_df$use_location = as.logical(debug_marginals_df$use_location)

tmp_debug = debug_marginals_df %>% 
  filter(experiment == "e") %>% 
  pivot_longer(6:11, names_to = "model", values_to = "marginal") %>% 
  filter(model %in% c("truth", "wellspec", "misspec", "NPMLE"))

tmp_debug %>%
  filter(variance == 0.1) %>%
  ggplot(aes(x = Z, y = marginal, color = model)) +
  geom_line() +
  facet_wrap(use_location ~ use_scale) +
  labs(title= "variance = 0.1")

tmp_debug %>%
  filter(variance == 0.5) %>%
  ggplot(aes(x = Z, y = marginal, color = model)) +
  geom_line() +
  facet_wrap(use_location ~ use_scale) +
  labs(title= "variance = 0.5")

marginals_med_scale_thin = marginals_med_scale[seq(1, dim(marginals_med_scale)[1], 5), ]

e_marginals_thin = marginals_med_scale_thin %>% 
  filter(experiment == "e") %>%
  mutate(variance = factor(variance))


NPMLE_misspec_truth = dark2_palette[c(8, 3, 1)]
NPMLE_misspec_truth[1] = "#2E2E2E"
NPMLE_misspec_truth = setNames(NPMLE_misspec_truth, rev(c("NPMLE", "SURE-PM", "Ground truth")))



levels(e_marginals_thin$variance) = c(TeX(r"($\sigma_i^2 = 0.1$)"),
                                      TeX(r"($\sigma_i^2 = 0.5$)"))


### Experiment (e) ####
# "true marginal", "marginal by SURE-PM (the one without covariates)", and "marginal by NPMLE"
e_marginals_thin %>% 
  filter(model_raw %in% c("truth", "misspec", "NPMLE")) %>%
  ggplot(aes(x=Z, y = marginal)) +
  geom_line(aes(color=model, linetype=model, alpha=(model!="Ground truth")), size=0.8) +
  facet_wrap( ~ variance, scales = "free", labeller=label_parsed) +
  scale_color_manual(values=NPMLE_misspec_truth, name="", breaks=c("NPMLE", "SURE-PM", "Ground truth")) +
  theme(legend.position="bottom", text=element_text(size=13),
        legend.key.size = unit(3,"line")) +
  ylab(TeX(r"($f_G(z \, | \, sigma_i^2)$)")) +
  xlab(TeX("z")) +
  scale_linetype_manual(values=c(2, 4, 1), name="", breaks=c("NPMLE", "SURE-PM", "Ground truth")) +
  guides(alpha="none") +
  scale_alpha_manual(values=c(1, 1))