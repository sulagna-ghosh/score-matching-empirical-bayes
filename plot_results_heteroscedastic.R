library(tidyverse)
library(ggh4x) # facetted_pos_scales
library(ggpubr) # arrange multiple figures together
library(latex2exp)
library(RColorBrewer)
library(lemon)
library(scales)
palette1 <- brewer.pal(8, "Dark2")
palette2 <- brewer.pal(8, "Set1")
palette3 <- brewer.pal(8, "Set2")
dark2_palette <- unique(c(palette1, palette2, palette3)) 
theme_set(theme_bw())

## Read and process in-sample MSEs ##### 
df_names = list.files("results/heteroscedastic")
df_names = df_names[str_detect(df_names, "experiment")]

df = NULL
for (df_name in df_names){
  experiment_str = substr(df_name, 12, 12)
  if (is.null(df)){
    df = read.csv(paste("results/heteroscedastic", df_name, sep="/"))
    df$experiment = experiment_str
  } else {
    df = df %>% bind_rows(read.csv(paste("results/heteroscedastic", df_name, sep="/")) 
                          %>% mutate(experiment=experiment_str))
  }
  
}

df = df %>% 
  select(-X) %>% 
  pivot_longer(5:17, names_to = "model", values_to = "value") %>% 
  separate(model, c("objective", "model"))

df$use_location = as.logical(df$use_location)
df$use_scale = as.logical(df$use_scale)

df_mean = df %>%
  group_by(n, use_location, use_scale, experiment, objective, model) %>%
  summarize(mean = mean(value, na.rm=T),
            count = n(),
            se = sd(value, na.rm=T)/sqrt(count), .groups="drop") %>% 
  mutate(se = if_else(n > 400, NA, se))


# Figure 1: in-sample MSE  ####

m_sim = unique(df_mean$count)

bayes_risk = data.frame("experiment" = c("c", "d", "e", "f", "g", "h", "i", "j"),
                        "yintercept" = c(0, 0, 0.15, 0, 0.036, 0.338, 1.327, 0.833),
                        "model" = rep("Bayes risk", 8)) %>%
  mutate(experiment = factor(experiment))

# calculated in miscellaneous/bayes_risk_calculation.ipynb


df_mean_plot = df_mean %>%
  filter(objective=="MSE",
         !use_location, use_scale, experiment != "d5",
         model != "NPMLEinit") %>%
  mutate(model = case_when(model == "misspec" ~ "SURE-PM",
                           model == "wellspec" ~ "SURE-THING",
                           model == "thetaG" ~ "SURE-grandmean",
                           model == "surels" ~ "SURE-LS", 
                           TRUE ~ model),
         model = factor(model, levels=c("NPMLE", "SURE-PM", "SURE-grandmean", "SURE-THING", "EBCF", "SURE-LS", "Bayes risk"))) %>%
  mutate(experiment = factor(experiment),
         experiment = factor(experiment, levels=c("c", "d", "e", "f", "g", "h", "i", "j"))) 

ground_truth_df = df_mean_plot %>% 
  group_by(n, experiment) %>%
  summarize(count = n()) %>% 
  left_join(bayes_risk, by = "experiment") %>%
  mutate(model = factor(model, levels=c("NPMLE", "SURE-PM", "SURE-grandmean", "SURE-THING", "EBCF", "SURE-LS", "Bayes risk")),
         experiment = factor(experiment)) %>%
  rename(mean = yintercept)


levels(df_mean_plot$experiment) = c(c = TeX(r"(Uniform$$ prior)"), 
                                    d = TeX(r"(Inverse-$\chi^2$ prior)"),
                                    e = TeX(r"(Bimodal $\mu_i$, Two-point $\sigma_i$)"),
                                    f = TeX(r"(Uniform$$ likelihood)"),
                                    g = TeX(r"(Two-point $\mu_i$, Uniform $\sigma_i$)"),
                                    h = TeX(r"(Poisson $\mu_i$)"),
                                    i = TeX(r"(Five$$ covariates)"),
                                    j = TeX(r"(Uni-covariate$$ heteroscedastic$$ prior)"))

levels(ground_truth_df$experiment) = c(c = TeX(r"(Uniform$$ prior)"), 
                                       d = TeX(r"(Inverse-$\chi^2$ prior)"),
                                       e = TeX(r"(Bimodal $\mu_i$, Two-point $\sigma_i$)"),
                                       f = TeX(r"(Uniform$$ likelihood)"),
                                       g = TeX(r"(Two-point $\mu_i$, Uniform $\sigma_i$)"),
                                       h = TeX(r"(Poisson $\mu_i$)"),
                                       i = TeX(r"(Five$$ covariates)"),
                                       j = TeX(r"(Uni-covariate$$ heteroscedastic$$ prior)"))

levels(bayes_risk$experiment) = c(c = TeX(r"(Uniform$$ prior)"), 
                                  d = TeX(r"(Inverse-$\chi^2$ prior)"),
                                  e = TeX(r"(Bimodal $\mu_i$, Two-point $\sigma_i$)"),
                                  f = TeX(r"(Uniform$$ likelihood)"),
                                  g = TeX(r"(Two-point $\mu_i$, Uniform $\sigma_i$)"),
                                  h = TeX(r"(Poisson $\mu_i$)"),
                                  i = TeX(r"(Five$$ covariates)"),
                                  j = TeX(r"(Uni-covariate$$ heteroscedastic$$ prior)"))

df_mean_plot_line = df_mean_plot %>%
  select(n, experiment, count, mean, model) %>%
  bind_rows(ground_truth_df)

ggplot(df_mean_plot, aes(x = factor(n), color=model, shape=model, linetype=model)) +
  geom_point(aes(y = mean), size = 2.5) +
  geom_line(data= df_mean_plot_line, aes(y = mean, group = model,  linetype=model), linewidth=0.75) +
  geom_errorbar(aes(ymin = mean-1.96*se, 
                    ymax = mean+1.96*se, 
                    group=model), alpha=0.5, show.legend=F, width=0.5) +
  facet_wrap( ~ experiment, scales="free_y", labeller=label_parsed, nrow = 4, ncol = 2) +
  # labs(title="In-sample MSE, scale only") +
  theme(axis.text.x = element_text(angle=45, hjust=1),
        legend.position="bottom",
        text=element_text(size=25),
        legend.key.size = unit(2,"line")) +
  geom_hline(data = bayes_risk, aes(yintercept = yintercept), show.legend = F) +
  geom_text(data = bayes_risk, aes(x = 5, y = yintercept, label = paste0("Bayes risk: ", yintercept)), hjust = -0.15, vjust = 1.35, color = 'black', size = 7) +
  scale_y_continuous(expand = expansion(mult = c(0.13, 0.1))) + 
  scale_color_manual(values=dark2_palette[c(1,3,2,4,10,9,8)], name="") +
  scale_linetype_manual(values=c(2, 4, 3, 1, 6, 5, 1), name="") +
  scale_shape_manual(values=c(15, 17, 16, 18, 19, 20, 1), name="") +
  ylab("In-sample MSE") +
  xlab("n")

ggsave("results/figures/figure_1.png", height=20, width=16) 

### Bimodal prior, two-point variance: One panel of MSEs #####

# Could you redo the panel for "bimodal prior with two-point variance" 
# including only SURE-grandmean, NPMLE, SURE-PM?
# (As a single panel and perhaps with thicker lines)

bimodal_only_mse_plot = df_mean_plot %>%
  filter(experiment == '"Bimodal " * mu[i] * ", Two-point " * sigma[i]',
         model %in% c('SURE-grandmean', 'NPMLE', 'SURE-PM')) %>%
  mutate(model = factor(model, levels = c("NPMLE","SURE-PM","SURE-grandmean")))

bimodel_only_bayes_risk = bayes_risk %>%
  filter(experiment == '"Bimodal " * mu[i] * ", Two-point " * sigma[i]')

ggplot() + 
  geom_point(data=bimodal_only_mse_plot,
             aes(x=factor(n), y=mean, color=model, shape=model),
             size=2.5) + 
  geom_line(data=bimodal_only_mse_plot,
            aes(x=factor(n), y=mean, group=model, color=model, linetype=model),
            linewidth=1) + 
  geom_errorbar(data=bimodal_only_mse_plot,
                aes(x=factor(n), group=model,
                    ymin=mean-1.96*se, ymax=mean+1.96*se,
                    color=model),
                alpha=0.5, show.legend=F, width=0.5) +
  theme(axis.text.x = element_text(angle=45, hjust=1),
        legend.position="bottom",
        text=element_text(size=25),
        legend.key.size = unit(2,"line")) +
  scale_y_continuous(expand = expansion(mult = c(0.13, 0.1))) + 
  scale_color_manual(values=dark2_palette[c(1,3,2,4,10,9,8)], name="") +
  scale_linetype_manual(values=c(2, 4, 3, 1, 6, 5, 1), name="") +
  scale_shape_manual(values=c(15, 17, 16, 18, 19, 20, 1), name="") +
  scale_y_continuous(limits = c(0.15, 0.3)) +
  #geom_hline(data = bimodel_only_bayes_risk, aes(yintercept = yintercept), show.legend = F) +
  # geom_text(data = bimodel_only_bayes_risk,
  #           aes(x = 5, y = yintercept, label = paste0("Bayes risk: ", yintercept)),
  #           show.legend=F,
  #           hjust = -0.15, vjust = 1.35, color = 'black', size = 7) +
  ylab("In-sample MSE") +
  xlab("n")

ggsave("results/figures/figure_1_bimodal_only.png", height=6, width=8) 

### Uni-covariate heteroscedastic prior: One panel of MSEs #####


unicovariate_only_mse_plot = df_mean_plot %>%
  filter(experiment == '"Uni-covariate" * " heteroscedastic" * " prior"',
         model != 'EBCF') %>%
  mutate(model = factor(model, levels = c("NPMLE","SURE-PM","SURE-grandmean", "SURE-THING",
                                          "SURE-LS")))

ggplot() + 
  geom_point(data=unicovariate_only_mse_plot,
             aes(x=factor(n), y=mean, color=model, shape=model),
             size=2.5) + 
  geom_line(data=unicovariate_only_mse_plot,
            aes(x=factor(n), y=mean, group=model, color=model, linetype=model),
            linewidth=1) + 
  geom_errorbar(data=unicovariate_only_mse_plot,
                aes(x=factor(n), group=model,
                    ymin=mean-1.96*se, ymax=mean+1.96*se,
                    color=model),
                alpha=0.5, show.legend=F, width=0.5) +
  theme(axis.text.x = element_text(angle=45, hjust=1),
        legend.position="bottom",
        text=element_text(size=25),
        legend.key.size = unit(2,"line")) +
  scale_y_continuous(expand = expansion(mult = c(0.13, 0.1))) + 
  scale_color_manual(values=dark2_palette[c(1,3,2,4,10,9,8)], name="") +
  scale_linetype_manual(values=c(2, 4, 3, 1, 6, 5, 1), name="") +
  scale_shape_manual(values=c(15, 17, 16, 18, 19, 20, 1), name="") +
  #geom_hline(data = bimodel_only_bayes_risk, aes(yintercept = yintercept), show.legend = F) +
  # geom_text(data = bimodel_only_bayes_risk,
  #           aes(x = 5, y = yintercept, label = paste0("Bayes risk: ", yintercept)),
  #           show.legend=F,
  #           hjust = -0.15, vjust = 1.35, color = 'black', size = 7) +
  ylab("In-sample MSE") +
  xlab("n") +
  guides(color=guide_legend(nrow=2,byrow=TRUE))

ggsave("results/figures/figure_1_unicovariate_only.png", height=6.5, width=8) 


# Read path for Figures 2, 3 ####

read_path = paste("results/one_run_heteroscedastic/") 

## Figure 2: priors for experiment (e) ####

prior_e =  read.csv(paste(read_path, "prior.csv", sep="")) %>%
  select(-X) %>%
  select(ends_with("e_median_scale")) %>% 
  pivot_longer(1:6) %>%
  mutate(prior_variable = case_when(str_detect(name, "theta_grid") ~ "theta",
                                    TRUE ~ "pi"),
         experiment = case_when(str_detect(name, "_c_") ~ "c",
                                str_detect(name, "_d_") ~ "d",
                                str_detect(name, "_d5_") ~ "d5",
                                str_detect(name, "_e_") ~ "e",
                                str_detect(name, "_f_") ~ "f",),
         model = case_when(str_detect(name, "misspec") ~ "misspec",
                           str_detect(name, "NPMLEinit") ~ "NPMLEinit",
                           str_detect(name, "NPMLE") ~ "NPMLE")) %>%
  select(-name) %>%
  pivot_wider(names_from = prior_variable, values_from = value, values_fn = list) %>%
  unnest(cols=c(theta, pi)) %>%
  filter(experiment == "e", model != "NPMLEinit") %>%
  mutate(model = case_when(model == "misspec" ~ "SURE-PM",
                           model == "wellspec" ~ "SURE-THING",
                           model == "thetaG" ~ "SURE-grandmean",
                           TRUE ~ model), 
         model = factor(model, levels = c("NPMLE","SURE-PM","SURE-grandmean")),
         plotting_trick = factor(model,levels = c("NPMLE","SURE-PM","SURE-grandmean")))

theta_truth_01 = seq(min(prior_e$theta), max(prior_e$theta), by = 0.01)
theta_truth_05 = seq(min(prior_e$theta), max(prior_e$theta), by = 0.01)
pi_truth_01 = dnorm(theta_truth_01, mean=2, sd=sqrt(0.1)) # low variance
pi_truth_05 = dnorm(theta_truth_05, mean=0, sd=sqrt(0.5)) # high variance
ground_truth_e =  data.frame('experiment' = rep("e", 2*length(theta_truth_01)),
                             'model' = factor(c(rep("low", length(theta_truth_01)),
                                                rep("high", length(theta_truth_01)))),
                             theta=c(theta_truth_01, theta_truth_05),
                             pi=c(pi_truth_01, pi_truth_05))

levels(ground_truth_e$model) = c(low = TeX(r"(T $\sigma^2_i = 0.1$)"),
                                 high = TeX(r"(T $\sigma^2_i = 0.5$)"))



levels(ground_truth_e$model) = c(low = TeX(r"($\sigma^2_i = 0.1$)"),
                                 high = TeX(r"($\sigma^2_i = 0.5$)"))

prior_e %>%
  ggplot() +
  geom_line(data=ground_truth_e, aes(x = theta, y = pi, group = model, linetype=model), size=0.8) +
  geom_point(aes(x = theta, y = pi, color=model), size=2, alpha=0.9)  + 
  geom_segment(aes(x = theta, y = pi, color=model, xend=theta,yend=pi - pi), alpha=0.9, size = 1) + 
  facet_wrap(~ plotting_trick, labeller = label_parsed) +
  scale_color_manual(values=c(dark2_palette[c(1, 3)], "#2E2E2E",  "#2E2E2E"), name="") +
  scale_linetype_manual(values=c("dotted", "dashed"), name="", labels = parse_format()) +
  theme(legend.position="bottom",
        text=element_text(size=24), legend.margin=margin(),
        legend.key.size = unit(3,"line"),
        axis.title.x = element_text(vjust=-1)) +
  xlab(TeX("$u_j$")) +
  ylab(TeX("$pi_j")) +
  scale_x_continuous(limit = c(-5, 5)) 

ggsave("results/figures/figure_2.png", height=6, width=10)

# Read and process shrinkage rules ####

xie_shrinkage = read.csv(paste(read_path, "shrinkage_rule.csv", sep="")) %>% 
  select(-X) %>%
  rename(thetaG = parametric_G, 
         misspec_median_scale = EB_misspec_median_scale,
         wellspec_median_scale = EB_wellspec_median_scale) %>%
  select(experiment, Z, variance, NPMLE, thetaG, misspec_median_scale, wellspec_median_scale, truth) %>%
  pivot_longer(4:8, names_to = "model", values_to = "posterior_mean") %>%
  mutate(experiment = factor(experiment)) %>%
  mutate(model_raw = model,
         model = case_when(model == "misspec_median_scale" ~ "SURE-PM",
                           model == "wellspec_median_scale" ~ "SURE-THING",
                           model == "thetaG" ~ "SURE-grandmean",
                           model == "truth" ~ "Ground truth",
                           TRUE ~ model),
         model = factor(model, levels = rev(c("NPMLE", "SURE-grandmean", "SURE-PM", "SURE-THING", "Ground truth"))),
         experiment = factor(experiment, levels=c("c", "d", "f", "e"))) 


xie_shrinkage_e = xie_shrinkage %>% 
  filter(experiment == "e")  %>%
  mutate(variance = factor(variance)) %>% 
  arrange(model)

xie_shrinkage_e_thin = xie_shrinkage_e[seq(1, dim(xie_shrinkage_e)[1], 5), ]


levels(xie_shrinkage_e_thin$variance) = c(TeX(r"($\sigma_i^2 = 0.1$)"),
                                          TeX(r"($\sigma_i^2 = 0.5$)"))

NPMLE_misspec_truth = dark2_palette[c(8, 3, 1)]
NPMLE_misspec_truth[1] = "#9E9E9E"
NPMLE_misspec_truth = setNames(NPMLE_misspec_truth, rev(c("NPMLE", "SURE-PM", "Ground truth")))

# Read and process marginals #### 

marginals_df = read.csv(paste(read_path,"marginal.csv", sep="")) %>% 
  select(-X) 

# 6400 n x 4 location scale x 5 experments x 3 variances
marginals_df$use_scale = as.logical(marginals_df$use_scale)
marginals_df$use_location = as.logical(marginals_df$use_location)

marginals_med_scale = marginals_df %>% 
  filter(use_scale, !use_location) %>%
  filter(experiment != "d5") %>% 
  pivot_longer(6:12, names_to = "model", values_to = "marginal") %>% 
  filter(model != 'NPMLEinit') %>%
  arrange(model) %>%
  mutate(experiment = factor(experiment)) %>%
  mutate(model_raw = model,
         model = case_when(model == "misspec" ~ "SURE-PM",
                           model == "wellspec" ~ "SURE-THING",
                           model == "thetaG" ~ "SURE-grandmean",
                           model == "truth" ~ "Ground truth",
                           model == "surels" ~ "SURE-LS",
                           TRUE ~ model),
         # switch the order of models so that ground truth is in the back of the plot
         model = factor(model, levels = c("Ground truth", "SURE-THING", "SURE-PM", "SURE-grandmean",  "NPMLE" )),
         experiment = factor(experiment, levels=c("c", "d", "f", "e")))

marginals_med_scale_thin = marginals_med_scale[seq(1, dim(marginals_med_scale)[1], 5), ]

e_marginals_thin = marginals_med_scale_thin %>% 
  filter(experiment == "e") %>%
  mutate(variance = factor(variance))

levels(e_marginals_thin$variance) = c(TeX(r"($\sigma_i^2 = 0.1$)"),
                                      TeX(r"($\sigma_i^2 = 0.5$)"))


### Find quantiles of marginal ##### 

alpha = 0.025

find_alpha_quantiles_of_marginal = function(marginal_and_shrinkage_df, alpha=0.025){
  
  marginal = marginal_and_shrinkage_df %>%
    filter(name == "B",  model == "Ground truth")  %>% # marginal
    arrange(Z) %>% # order by Z
    pull(value)
  
  Z = marginal_and_shrinkage_df %>%
    filter(name == "B",  model == "Ground truth")  %>% # marginal
    arrange(Z) %>% # order by Z
    pull(Z)
  
  normalising_constant = sum(marginal)
  
  alpha_idx = max(which(cumsum(marginal)/normalising_constant <= alpha))
  one_minus_alpha_idx = min(which(cumsum(marginal)/normalising_constant >= 1 - alpha))
  
  return(c(Z[alpha_idx], Z[one_minus_alpha_idx]))
  
}

# Process shrinkage and marginal for low variance component, 0.1 ####

marginal_and_shrinkage_01 = xie_shrinkage_e_thin %>%
  filter(variance =="sigma[i]^{\n    2\n} * {\n    phantom() == phantom()\n} * 0.1") %>%
  filter(model_raw %in% c("truth", "misspec_median_scale", "NPMLE")) %>%
  rename(value=posterior_mean) %>%
  mutate(name="A") %>%
  bind_rows(e_marginals_thin %>% 
              select(-use_location, -use_scale) %>%
              filter(variance =="sigma[i]^{\n    2\n} * {\n    phantom() == phantom()\n} * 0.1") %>%
              filter(model_raw %in% c("truth", "misspec", "NPMLE")) %>%
              rename(value=marginal) %>%
              mutate(name="B"))

find_alpha_quantiles_of_marginal(marginal_and_shrinkage_01)

ribbon_marginal_01 = data.frame(Z = rep(find_alpha_quantiles_of_marginal(marginal_and_shrinkage_01, alpha=0.025), 1),
                                y = c(1, 1),
                                name = c("B", "B"))

marginal_and_shrinkage_01 = marginal_and_shrinkage_01 %>%
  mutate(alpha_Z_groundtruth = ribbon_marginal_01$Z[1],
         one_minus_alpha_Z_grountruth = ribbon_marginal_01$Z[2],
         alpha_95 = case_when(name == "A" & Z > one_minus_alpha_Z_grountruth ~ "transparency1", 
                              name == "A" & Z < alpha_Z_groundtruth ~ "transparency2", 
                              name == "A" ~ "solid",
                              TRUE ~ NA))

# LHS of Figure 3 (low variance component, 0.1) #### 

ggplot() +
  geom_area(data=ribbon_marginal_01, aes(x=Z, y=y), fill="#9EEEEE", alpha=0.4) +
  geom_line(data=marginal_and_shrinkage_01 %>%
              mutate(delete_to_truncate = (name == "A" & value < 1)) %>% 
              filter(Z >= -3, Z <= 3, !delete_to_truncate), aes(x=Z, y = value, color=model, linetype=model, alpha=alpha_95), size=0.8) +
  facet_rep_grid(rows= vars(name), scales="free_y",
                 labeller = as_labeller(c(B = "Marginal", `A` = "Posterior mean") ),
                 switch="both",
                 repeat.tick.labels = T) +
  scale_color_manual(values=NPMLE_misspec_truth, name="", breaks=c("NPMLE", "SURE-PM", "Ground truth")) +
  theme(legend.position="bottom", text=element_text(size=13),
        legend.key.size = unit(3,"line"),
        strip.background = element_blank(),
        strip.placement = 'outside') +
  ylab(TeX(r"($f_G(z \, | \, sigma_i^2)$)")) +
  xlab(TeX("z")) +
  scale_linetype_manual(values=c(2, 4, 1), name="", breaks=c("NPMLE", "SURE-PM", "Ground truth")) +
  guides(alpha="none", linetype="none", color = "none") +
  scale_alpha_manual(values=c(1, .5, .5)) +
  ylab("")

ggsave("results/figures/figure_3_left.png", width = 4, height=5)

# Process shrinkage and marginal for high variance component, 0.5 ####

marginal_and_shrinkage_05 = xie_shrinkage_e_thin %>%
  filter(variance =="sigma[i]^{\n    2\n} * {\n    phantom() == phantom()\n} * 0.5") %>%
  filter(model_raw %in% c("truth", "misspec_median_scale", "NPMLE")) %>%
  rename(value=posterior_mean) %>%
  mutate(name="A") %>%
  bind_rows(e_marginals_thin %>% 
              select(-use_location, -use_scale) %>%
              filter(variance =="sigma[i]^{\n    2\n} * {\n    phantom() == phantom()\n} * 0.5") %>%
              filter(model_raw %in% c("truth", "misspec", "NPMLE")) %>%
              rename(value=marginal) %>%
              mutate(name="B"))

ribbon_marginal_05 = data.frame(Z = find_alpha_quantiles_of_marginal(marginal_and_shrinkage_05, alpha=0.025),
                                y = c(1, 1),
                                name = c("B", "B"))

marginal_and_shrinkage_05 = marginal_and_shrinkage_05 %>%
  mutate(alpha_Z_groundtruth = ribbon_marginal_05$Z[1],
         one_minus_alpha_Z_grountruth = ribbon_marginal_05$Z[2],
         alpha_95 = case_when( name == "A" & Z > one_minus_alpha_Z_grountruth ~ "transparency1", 
                               name == "A" & Z < alpha_Z_groundtruth ~ "transparency2", 
                               name == "A" ~ "solid",
                               TRUE ~ NA))

# RHS of Figure 3 ####

ggplot() +
  geom_area(data=ribbon_marginal_05, aes(x=Z, y=y), fill="#9EEEEE", alpha=0.4) +
  geom_line(data=marginal_and_shrinkage_05  %>%
              mutate(delete_to_truncate = (name == "A" & value < -1.5)) %>% 
              filter(Z >= -3, Z <= 3, !delete_to_truncate), aes(x=Z, y = value, color=model, linetype=model, alpha=alpha_95), size=0.8) +
  facet_rep_grid(rows= vars(name), scales="free_y",
                 labeller = as_labeller(c(B = "Marginal", A = "Posterior mean") ),
                 switch="both",
                 repeat.tick.labels = T) +
  scale_color_manual(values=NPMLE_misspec_truth, name="", breaks=c("NPMLE", "SURE-PM", "Ground truth")) +
  theme(legend.position="bottom", text=element_text(size=13),
        legend.key.size = unit(3,"line"),
        strip.background = element_blank(),
        strip.placement = 'outside') +
  ylab(TeX(r"($f_G(z \, | \, sigma_i^2)$)")) +
  xlab(TeX("z")) +
  scale_linetype_manual(values=c(2, 4, 1), name="", breaks=c("NPMLE", "SURE-PM", "Ground truth")) +
  guides(alpha="none",linetype="none", color = "none") +
  scale_alpha_manual(values=c(1, .25, .25)) +
  ylab("") +
  geom_point(data=data.frame(x = c(0, 0), y = c(-1, 3), name= "A"),
             aes(x=x, y=y), alpha=0)

ggsave("results/figures/figure_3_right.png", width = 4, height=5) 

# Manually combine left and right

# Bottom two panels of Figure 3 ####

marginal_01_and_05 = marginal_and_shrinkage_01 %>%
  filter(name == "B") %>%
  bind_rows( (marginal_and_shrinkage_05 %>%
              filter(name == "B")) )

ribbon_marginal_01_and_05 = ribbon_marginal_01 %>%
  bind_rows(ribbon_marginal_05) %>%
  bind_cols("variance" = c("sigma[i]^{\n    2\n} * {\n    phantom() == phantom()\n} * 0.1",
              "sigma[i]^{\n    2\n} * {\n    phantom() == phantom()\n} * 0.1",
              "sigma[i]^{\n    2\n} * {\n    phantom() == phantom()\n} * 0.5",
              "sigma[i]^{\n    2\n} * {\n    phantom() == phantom()\n} * 0.5"))


ggplot() +
  geom_area(data=ribbon_marginal_01_and_05, aes(x=Z, y=y), fill="#9EEEEE", alpha=0.4) +
  geom_line(data=marginal_01_and_05  %>%
              mutate(delete_to_truncate = (name == "A" & value < -1.5)) %>% 
              filter(Z >= -3, Z <= 3, !delete_to_truncate), aes(x=Z, y = value, color=model, linetype=model, alpha=alpha_95), size=0.8) +
  facet_rep_grid(cols= vars(variance), scales="free_y",
                 switch="both",
                 repeat.tick.labels = T) +
  scale_color_manual(values=NPMLE_misspec_truth, name="", breaks=c("NPMLE", "SURE-PM", "Ground truth")) +
  theme(legend.position="bottom", text=element_text(size=13),
        legend.key.size = unit(3,"line"),
        strip.background = element_blank(),
        strip.placement = 'outside',
        strip.text=element_blank()) +
  ylab(TeX(r"($f_G(z \, | \, sigma_i^2)$)")) +
  xlab(TeX("z")) +
  scale_linetype_manual(values=c(2, 4, 1), name="", breaks=c("NPMLE", "SURE-PM", "Ground truth")) +
  guides(alpha="none",linetype="none", color = "none") +
  scale_alpha_manual(values=c(1, .25, .25)) +
  ylab("Marginal\n") 

ggsave("results/figures/figure_3_bottom.png", width = 12, height=3) 

# Top two panels of Figure 3 ####

shrinkage_01_and_05 = marginal_and_shrinkage_01 %>%
  filter(name == "A") %>%
  bind_rows( (marginal_and_shrinkage_05 %>%
                filter(name == "A")) )

invisible_point_for_scaling = data.frame(x = c(0, 0), y = c(-1, 3), name= "A",
                                         variance = c("sigma[i]^{\n    2\n} * {\n    phantom() == phantom()\n} * 0.5",
                                                      "sigma[i]^{\n    2\n} * {\n    phantom() == phantom()\n} * 0.5")) %>%
  mutate(variance = factor(variance, levels = c("sigma[i]^{\n    2\n} * {\n    phantom() == phantom()\n} * 0.1",
                                                "sigma[i]^{\n    2\n} * {\n    phantom() == phantom()\n} * 0.5")))


ggplot() +
  geom_line(data=shrinkage_01_and_05  %>%
              mutate(delete_to_truncate = (name == "A" & value < -1.5)) %>% 
              filter(Z >= -3, Z <= 3, !delete_to_truncate, 
                     !((variance == "sigma[i]^{\n    2\n} * {\n    phantom() == phantom()\n} * 0.1") & value < 1.25)), 
            aes(x=Z, y = value, color=model, linetype=model, alpha=alpha_95), size=0.8) +
  geom_point(data=invisible_point_for_scaling,
             aes(x=x, y=y, group=variance), alpha=0) +
  facet_wrap(~ variance, scales="free") +
  scale_color_manual(values=NPMLE_misspec_truth, name="", breaks=c("NPMLE", "SURE-PM", "Ground truth")) +
  theme(legend.position="bottom", text=element_text(size=13),
        legend.key.size = unit(3,"line"),
        strip.background = element_blank(),
        strip.placement = 'outside',
        strip.text=element_blank()) +
  ylab(TeX(r"($f_G(z \, | \, sigma_i^2)$)")) +
  xlab(TeX("z")) +
  scale_linetype_manual(values=c(2, 4, 1), name="", breaks=c("NPMLE", "SURE-PM", "Ground truth")) +
  guides(alpha="none",linetype="none", color = "none") +
  scale_alpha_manual(values=c(1, .25, .25)) +
  ylab("Posterior mean\n") +
  scale_x_continuous(limits = c(-3, 3)) +
  xlab("")

ggsave("results/figures/figure_3_top.png", width = 12, height=3) 

# Check the SURE of NPMLE vs. the SURE of SURE-PM ####


npmle_and_misspec_df = df %>% 
  select(SURE_NPMLE, SURE_misspec, experiment, n) %>%
  filter(!is.na(SURE_NPMLE) & !is.na(SURE_misspec)) %>%
  mutate(is_sure_of_pm_smaller = SURE_misspec < SURE_NPMLE) 

failed_experiments = npmle_and_misspec_df %>% 
  filter(!is_sure_of_pm_smaller) %>% select(experiment, n) %>% unique()

npmle_and_misspec_df %>%
  group_by(n, experiment, is_sure_of_pm_smaller) %>%
  summarize(count = n(), 
            pct = round(n() / dim(npmle_and_misspec_df)[1], 5)) %>% view()

npmle_and_misspec_df %>%
  group_by(is_sure_of_pm_smaller) %>%
  summarize(count = n(), 
            pct = n() / dim(npmle_and_misspec_df)[1])

npmle_and_misspec_df %>% 
  filter((experiment %in% failed_experiments$experiment ) & 
           n %in% failed_experiments$n) %>%
  ggplot(aes(x = SURE_misspec, y = SURE_NPMLE, color = is_sure_of_pm_smaller)) + 
           geom_point() + facet_wrap(experiment ~ n, scales="free", nrow=3) + 
           geom_abline() +
  theme(legend.position = "bottom")

## Read data when setting seed ##### 

# NOTE: only for one experiment so far
df_names = list.files("results/heteroscedastic")
df_names = df_names[str_detect(df_names, "experiment_c")]

df = NULL
for (df_name in df_names){ 
  
  if (is.null(df)){
    df = read.csv(paste("results/heteroscedastic", df_name, sep="/"))
  } else {
    df = df %>% bind_rows(read.csv(paste("results/heteroscedastic", df_name, sep="/")))
  }
  
}

df_subset = df %>% 
  select(n, SURE_misspec, SURE_NPMLE) %>%
  mutate(NPMLE_PM_diff = SURE_NPMLE - SURE_misspec)

df_subset %>% group_by(n) %>% summarize(NPMLE_lower_objective = sum(NPMLE_PM_diff < 0))


## Optimization checks #####

df = read.csv("results/optimization_checks/c_200_different_inits.csv") %>% 
  bind_rows(read.csv("results/optimization_checks/d_200_different_inits.csv")) %>%
  bind_rows(read.csv("results/optimization_checks/e_200_different_inits.csv")) %>%
  bind_rows(read.csv("results/optimization_checks/f_200_different_inits.csv")) %>%
  bind_rows(read.csv("results/optimization_checks/g_200_different_inits.csv")) %>%
  bind_rows(read.csv("results/optimization_checks/h_200_different_inits.csv")) %>% 
  bind_rows(read.csv("results/optimization_checks/i_200_different_inits.csv")) %>%
  bind_rows(read.csv("results/optimization_checks/j_200_different_inits.csv"))

# Calculations

# sqrt( samplevariance( SURE of uniform init ) )
variation_across_init = df %>%
  filter(estimator == "SURE-PM uniform") %>%
  group_by(experiment) %>%
  summarize(sample_sd_sure_uniform = sqrt(var(sure)))

# sqrt( mean (samplevariance(SURE of random init within simulation) ))
random_variation_within_replicate = df %>%
  filter(estimator == "SURE-PM random") %>%
  group_by(experiment, m) %>%
  summarize(sample_var = var(sure), .groups="drop") %>%
  group_by(experiment) %>%
  summarize(sqrt_avg_sample_variance_random = sqrt( mean(sample_var )))

# sqrt( 1/(2R) sum( (SURE of NPMLEinit - SURE of uniform init)^2 ))
variation_of_NPMLE_and_uniform_init = df %>%
  filter(estimator %in% c("SURE-PM NPMLEinit", "SURE-PM uniform")) %>%
  select(estimator, sure, experiment, m) %>%
  pivot_wider(names_from=estimator, values_from=sure) %>%
  mutate(difference_sq = (`SURE-PM uniform` - `SURE-PM NPMLEinit`)**2 ) %>%
  group_by(experiment) %>%
  summarize(sqrt_mean_of_squared_difference_NPMLEinit_divided_by_2 = sqrt(mean(difference_sq)/2))

# sqrt( 1/(2R) sum( (SURE of random init - SURE of uniform init)^2 ))
# SURE of random init is an average
variation_of_random_and_uniform_init = df %>%
  filter(estimator %in% c("SURE-PM random", "SURE-PM uniform")) %>%
  select(estimator, sure, experiment, m) %>%
  group_by(estimator, experiment) %>%
  summarize(mean_sure = mean(sure), .groups="drop") %>%
  pivot_wider(names_from=estimator, values_from=mean_sure) %>%
  mutate(difference_sq = (`SURE-PM uniform` - `SURE-PM random`)**2 ) %>%
  group_by(experiment) %>%
  summarize(sqrt_mean_of_squared_difference_randominit_divided_by_2 = sqrt(mean(difference_sq)/2))

combined_metrics = variation_across_init %>%
  left_join(random_variation_within_replicate, by='experiment') %>%
  left_join(variation_of_NPMLE_and_uniform_init, by='experiment') %>%
  left_join(variation_of_random_and_uniform_init, by='experiment') %>%
  mutate(relative_variation_within_random_init = sqrt_avg_sample_variance_random / sample_sd_sure_uniform,
         relative_variation_of_NPMLE_and_uniform_init = sqrt_mean_of_squared_difference_NPMLEinit_divided_by_2 / sample_sd_sure_uniform,
         relative_variation_of_random_and_uniform_init = sqrt_mean_of_squared_difference_randominit_divided_by_2 / sample_sd_sure_uniform)


combined_metrics %>% select(experiment, relative_variation_within_random_init, 
                            relative_variation_of_NPMLE_and_uniform_init, 
                            relative_variation_of_random_and_uniform_init) %>%
  mutate(relative_variation_within_random_init = round(relative_variation_within_random_init, 8), 
         relative_variation_of_NPMLE_and_uniform_init = round(relative_variation_of_NPMLE_and_uniform_init, 8), 
         relative_variation_of_random_and_uniform_init= round(relative_variation_of_random_and_uniform_init, 8)) 


df  %>% 
  filter(estimator!= "NPMLE",
         experiment %in% c("h", "i", "j")) %>%
ggplot(aes(x = sure)) + 
  geom_histogram(aes(fill = estimator)) + 
  facet_grid(estimator~experiment, scales="free_x")


df  %>% 
  filter(estimator!= "NPMLE") %>%
  ggplot(aes(x = sure)) + 
  geom_histogram(aes(fill = estimator)) + 
  facet_grid(estimator~experiment, scales="free_x")


# Optimization checks different inits ####

df_hij = read.csv("results/optimization_checks/h_200_different_inits_niter8000_lr-4.csv") %>%
  bind_rows(read.csv("results/optimization_checks/i_200_different_inits_niter8000_lr-4.csv")) %>%
  bind_rows(read.csv("results/optimization_checks/j_200_different_inits_niter8000_lr-4.csv"))


# Calculations

# sqrt( samplevariance( SURE of uniform init ) )
variation_across_init_hij = df_hij %>%
  filter(estimator == "SURE-PM uniform") %>%
  group_by(experiment) %>%
  summarize(sample_sd_sure_uniform = sqrt(var(sure)))

# sqrt( mean (samplevariance(SURE of random init within simulation) ))
random_variation_within_replicate_hij = df_hij %>%
  filter(estimator == "SURE-PM random") %>%
  group_by(experiment, m) %>%
  summarize(sample_var = var(sure), .groups="drop") %>%
  group_by(experiment) %>%
  summarize(sqrt_avg_sample_variance_random = sqrt( mean(sample_var )))

# sqrt( 1/(2R) sum( (SURE of NPMLEinit - SURE of uniform init)^2 ))
variation_of_NPMLE_and_uniform_init_hij = df_hij %>%
  filter(estimator %in% c("SURE-PM NPMLEinit", "SURE-PM uniform")) %>%
  select(estimator, sure, experiment, m) %>%
  pivot_wider(names_from=estimator, values_from=sure) %>%
  mutate(difference_sq = (`SURE-PM uniform` - `SURE-PM NPMLEinit`)**2 ) %>%
  group_by(experiment) %>%
  summarize(sqrt_mean_of_squared_difference_NPMLEinit_divided_by_2 = sqrt(mean(difference_sq)/2))

# sqrt( 1/(2R) sum( (SURE of random init - SURE of uniform init)^2 ))
# SURE of random init is an average
variation_of_random_and_uniform_init_hij = df_hij %>%
  filter(estimator %in% c("SURE-PM random", "SURE-PM uniform")) %>%
  select(estimator, sure, experiment, m) %>%
  group_by(estimator, experiment) %>%
  summarize(mean_sure = mean(sure), .groups="drop") %>%
  pivot_wider(names_from=estimator, values_from=mean_sure) %>%
  mutate(difference_sq = (`SURE-PM uniform` - `SURE-PM random`)**2 ) %>%
  group_by(experiment) %>%
  summarize(sqrt_mean_of_squared_difference_randominit_divided_by_2 = sqrt(mean(difference_sq)/2))

combined_metrics_hij = variation_across_init_hij %>%
  left_join(random_variation_within_replicate, by='experiment') %>%
  left_join(variation_of_NPMLE_and_uniform_init, by='experiment') %>%
  left_join(variation_of_random_and_uniform_init, by='experiment') %>%
  mutate(relative_variation_within_random_init = sqrt_avg_sample_variance_random / sample_sd_sure_uniform,
         relative_variation_of_NPMLE_and_uniform_init = sqrt_mean_of_squared_difference_NPMLEinit_divided_by_2 / sample_sd_sure_uniform,
         relative_variation_of_random_and_uniform_init = sqrt_mean_of_squared_difference_randominit_divided_by_2 / sample_sd_sure_uniform)


combined_metrics_hij %>% select(experiment, relative_variation_within_random_init, 
                            relative_variation_of_NPMLE_and_uniform_init, 
                            relative_variation_of_random_and_uniform_init) %>%
  mutate(relative_variation_within_random_init = round(relative_variation_within_random_init, 8), 
         relative_variation_of_NPMLE_and_uniform_init = round(relative_variation_of_NPMLE_and_uniform_init, 8), 
         relative_variation_of_random_and_uniform_init= round(relative_variation_of_random_and_uniform_init, 8)) 


df_hij  %>% 
  filter(estimator!= "NPMLE") %>%
  ggplot(aes(x = sure)) + 
  geom_histogram(aes(fill = estimator)) + 
  facet_grid(estimator~experiment, scales="free_x")

