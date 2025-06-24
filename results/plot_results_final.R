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

## In-sample MSE ##### 

df_names = list.files("xie_losses/final_results")
df_names = df_names[str_detect(df_names, "location_scale_comparison")]

df = NULL
for (df_name in df_names[-1]){ # ignore location_scale_comparison 0
  
  if (is.null(df)){
    df = read.csv(paste("xie_losses/final_results", df_name, sep="/"))
  } else {
    df = df %>% bind_rows(read.csv(paste("xie_losses/final_results", df_name, sep="/")))
  }
  
}

df = df %>% 
  select(-X) %>% 
  pivot_longer(5:17, names_to = "model", values_to = "value") %>% 
  separate(model, c("objective", "model"))

df$use_location = as.logical(df$use_location)
df$use_scale = as.logical(df$use_scale)

df_mean = df %>%
  group_by(n, use_location, use_scale, experiment, data, objective, model) %>%
  summarize(mean = mean(value, na.rm=T),
            count = n(),
            se = sd(value, na.rm=T)/sqrt(count), .groups="drop")

df_mean = df_mean  %>% 
  mutate(se = if_else(n > 400, NA, se))

view(df_mean)

#### Plot: !use location, use scale ####

m_sim = unique(df_mean$count)

bayes_risk = data.frame("experiment" = c("c", "d", "e", "f", "g", "h", "i", "j"),
                        "yintercept" = c(0, 0, 0.15, 0, 0.036, 0.338, 1.327, 0.833),
                        "model" = rep("Bayes risk", 8)) %>%
  mutate(experiment = factor(experiment))

# bayes_risk = data.frame("experiment" = c("c", "d", "e", "f"),
#                         "yintercept" = c(0, 0, 0.15, 0),
#                         "model" = rep("Bayes risk", 4)) %>%
#   mutate(experiment = factor(experiment))


df_mean_plot = df_mean %>%
  filter(data=="train", objective=="MSE",
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
         # experiment = factor(experiment, levels=c("c", "d", "e", "f"))) 

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

# levels(df_mean_plot$experiment) = c(c = TeX(r"(Uniform$$ prior)"), 
#                                     d = TeX(r"(Inverse-$\chi^2$ prior)"), 
#                                     e = TeX(r"(Bimodal $\mu_i$, Two-point $\sigma_i$)"),
#                                     f = TeX(r"(Uniform$$ likelihood)"))
# 
# levels(ground_truth_df$experiment) = c(c = TeX(r"(Uniform$$ prior)"), 
#                                        d = TeX(r"(Inverse-$\chi^2$ prior)"), 
#                                        e = TeX(r"(Bimodal $\mu_i$, Two-point $\sigma_i$)"),
#                                        f = TeX(r"(Uniform$$ likelihood)"))
# 
# levels(bayes_risk$experiment) = c(c = TeX(r"(Uniform$$ prior)"), 
#                                   d = TeX(r"(Inverse-$\chi^2$ prior)"), 
#                                   e = TeX(r"(Bimodal $\mu_i$, Two-point $\sigma_i$)"),
#                                   f = TeX(r"(Uniform$$ likelihood)"))


df_mean_plot_line = df_mean_plot %>%
  select(n, experiment, count, mean, model,) %>%
  bind_rows(ground_truth_df)

# df_mean_plot = df_mean_plot %>% filter(model != 'Bayes risk')
# df_mean_plot_line = df_mean_plot_line %>% filter(model != 'Bayes risk')

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
  # geom_hline(data = bayes_risk, aes(yintercept = yintercept), show.legend = F) +
  geom_text(data = bayes_risk, aes(x = 5, y = yintercept, label = paste0("Bayes risk: ", yintercept)), hjust = -0.15, vjust = 1.35, color = 'black', size = 7) +
  scale_y_continuous(expand = expansion(mult = c(0.13, 0.1))) + 
  scale_color_manual(values=dark2_palette[c(1,3,2,4,10,9,8)], name="") +
  scale_linetype_manual(values=c(2, 4, 3, 1, 6, 5, 1), name="") +
  scale_shape_manual(values=c(15, 17, 16, 18, 19, 20, 1), name="") +
  ylab("In-sample MSE") +
  xlab("n")

ggsave("xie_plots_with_all/median_scale_MSE_all_final_small.png", height=20, width=16)

ggplot(df_mean_plot, aes(x = factor(n), color=model,shape=model)) +
  geom_point(aes(y = mean), size = 3) +
  geom_line(data= df_mean_plot_line, aes(y = mean, group = model,  linetype=model), size=0.8) +
  geom_errorbar(aes(ymin = mean-1.96*se, 
                    ymax = mean+1.96*se, 
                    group=model), alpha=0.5, show.legend=F) +
  facet_wrap( ~ experiment, scales="free_y", labeller=label_parsed, nrow = 4, ncol = 2) +
  # labs(title="In-sample MSE, scale only") +
  theme(axis.text.x = element_text(angle=45, hjust=1),
        legend.position="bottom",
        text=element_text(size=24),
        legend.key.size = unit(3,"line")) +
  scale_color_manual(values=dark2_palette[c(1:4,5,9,8)], name="") +
  scale_linetype_manual(values=c(2, 3, 4, 1, 5, 6, 1), name="") +
  scale_shape_manual(values=c(15, 16, 17, 18, 20, 19, 1), name="") +
  ylab("In-sample MSE") +
  xlab("n")

ggsave("xie_plots_with_all/median_scale_MSE_all_final_large.png", height=20, width=15)

## Shrinkage and marginals ##### 

seed_str = "60050100"
read_path = paste("xie_checks_", seed_str, "/", sep="")

# Shrinkage / posterior means ####

xie_shrinkage = read.csv(paste(read_path, "xie_shrinkage_location_scale.csv", sep="")) %>% 
  select(-X) %>%
  rename(thetaG = parametric_G, 
         misspec_median_scale = EB_misspec_median_scale,
         wellspec_median_scale = EB_wellspec_median_scale) %>%
  select(experiment, Z, variance, NPMLE, thetaG, misspec_median_scale, wellspec_median_scale, EB_surels, truth) %>%
  pivot_longer(4:9, names_to = "model", values_to = "posterior_mean") %>%
  mutate(experiment = factor(experiment)) %>%
  mutate(model_raw = model,
         model = case_when(model == "misspec_median_scale" ~ "SURE-PM",
                           model == "wellspec_median_scale" ~ "SURE-THING",
                           model == "thetaG" ~ "SURE-grandmean",
                           model == "EB_surels" ~ "SURE-LS", 
                           model == "truth" ~ "Bayes risk",
                           TRUE ~ model),
         model = factor(model, levels = rev(c("NPMLE", "SURE-grandmean", "SURE-PM", "SURE-THING", "SURE-LS", "Bayes risk"))),
         experiment = factor(experiment, levels=c("c", "d", "f", "e"))) 

# "truth", "SURE-LS", "SURE-PM" and NPMLE

xie_shrinkage_e = xie_shrinkage %>% 
  filter(experiment == "e")  %>%
  mutate(variance = factor(variance)) %>% 
  arrange(model)

xie_shrinkage_e_thin = xie_shrinkage_e[seq(1, dim(xie_shrinkage_e)[1], 5), ]


levels(xie_shrinkage_e_thin$variance) = c(TeX(r"($\sigma_i^2 = 0.1$)"),
                                          TeX(r"($\sigma_i^2 = 0.5$)"))

NPMLE_misspec_truth = dark2_palette[c(8, 3, 4, 1)]
NPMLE_misspec_truth[1] = "#9E9E9E9E"
NPMLE_misspec_truth = setNames(NPMLE_misspec_truth, rev(c("NPMLE", "SURE-PM", "SURE-LS", "Bayes risk")))

xie_shrinkage_e_thin %>%
  filter(model_raw %in% c("truth", "EB_surels", "misspec_median_scale", "NPMLE")) %>%
  ggplot(aes(x = Z, y = posterior_mean, color = model, linetype=model)) +
  geom_line(size=1.5) +
  facet_wrap(~ variance, scales = "free_x", labeller=label_parsed) +
  # geom_hline(data = data.frame(variance = c(0.1, 0.5),
  #                              theta = c(2, 0)), 
  #            aes(yintercept = theta)) +
  scale_color_manual(values = NPMLE_misspec_truth, name="", breaks=c("NPMLE", "SURE-PM", "SURE-LS", "Bayes risk")) +
  scale_linetype_manual(values=c(2, 4, 3, 1), name="", breaks=c("NPMLE", "SURE-PM", "SURE-LS", "Bayes risk")) +
  xlab(TeX("$z$")) +
  ylab("Posterior mean") +
  theme(legend.key.size = unit(3,"line"),
        text=element_text(size=24),
        axis.title.y=element_text(size=18))

ggsave("xie_plots_with_all/shrinkage_e_misspec_NPMLE_surels_median_scale.png", width=10, height=6)

full_palette = c(dark2_palette[c(1:5)], "#CCCCCC")
full_palette = setNames(full_palette, c("NPMLE", "SURE-grandmean", "SURE-PM", "SURE-THING", "SURE-LS", "Bayes risk"))

xie_shrinkage_e_thin %>% 
  ggplot(aes(x = Z, y = posterior_mean, color = model, linetype=model)) +
  geom_line(size=0.8) +
  facet_wrap(~ variance, scales = "free_x", labeller=label_parsed) +
  scale_color_manual(values = full_palette, name="", breaks = c("NPMLE", "SURE-grandmean", "SURE-PM", "SURE-THING", "SURE-LS", "Bayes risk")) +
  scale_linetype_manual(values=c(2, 3, 4, 1, 5, 1), name = "", breaks=c("NPMLE", "SURE-grandmean", "SURE-PM", "SURE-THING", "SURE-LS", "Bayes risk"))+
  xlab(TeX("$z$")) +
  ylab("Posterior mean") +
  theme(legend.key.size = unit(2,"line"))

ggsave("xie_plots_with_all/shrinkage_e_all_median_scale.png", width=6.5, height=4)

xie_shrinkage_e_thin %>% 
  ggplot(aes(x = Z, y = posterior_mean, color = model)) +
  geom_line(size=0.8) +
  facet_wrap(~ variance, scales = "free_x", labeller=label_parsed) +
  scale_color_manual(values = full_palette, name="", breaks = c("NPMLE", "SURE-grandmean", "SURE-PM", "SURE-THING", "SURE-LS", "Bayes risk")) +
  scale_linetype_manual(values=c(2, 3, 4, 1, 5, 1), name = "", breaks=c("NPMLE", "SURE-grandmean", "SURE-PM", "SURE-THING", "SURE-LS", "Bayes risk"))+
  xlab(TeX("$z$")) +
  ylab("Posterior mean") +
  theme(legend.key.size = unit(2,"line"))

ggsave("xie_plots_with_all/shrinkage_e_all_median_scale_solid.png", width=6.5, height=4)

# Marginals ####

read_path = "xie_checks_60050100/"

marginals_df = read.csv(paste(read_path,"location_scale_marginals.csv", sep="")) %>% 
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
                           model == "surels" ~ "SURE-LS", 
                           model == "truth" ~ "Bayes risk",
                           TRUE ~ model),
         # switch the order of models so that Bayes risk is in the back of the plot
         model = factor(model, levels = c("Bayes risk", "SURE-THING", "SURE-PM", "SURE-grandmean", "SURE-LS", "NPMLE" )),
         experiment = factor(experiment, levels=c("c", "d", "f", "e")))

marginals_med_scale_thin = marginals_med_scale[seq(1, dim(marginals_med_scale)[1], 5), ]

e_marginals_thin = marginals_med_scale_thin %>% 
  filter(experiment == "e") %>%
  mutate(variance = factor(variance))


NPMLE_misspec_truth = dark2_palette[c(8, 3, 4, 2, 1)]
NPMLE_misspec_truth[1] = "#9E9E9E9E"
NPMLE_misspec_truth = setNames(NPMLE_misspec_truth, rev(c("NPMLE", "SURE-PM", "SURE-THING", "SURE-LS", "Bayes risk")))

levels(e_marginals_thin$variance) = c(TeX(r"($\sigma_i^2 = 0.1$)"),
                                      TeX(r"($\sigma_i^2 = 0.5$)"))


tmp = marginals_df %>% 
  filter(experiment == "e") %>% 
  pivot_longer(6:12, names_to = "model", values_to = "marginal") %>% 
  filter(model %in% c("truth", "wellspec", "misspec", "surels", "NPMLE"))

tmp %>%
  filter(variance == 0.1) %>%
  ggplot(aes(x = Z, y = marginal, color = model)) +
  geom_line() +
  facet_wrap(use_location ~ use_scale) +
  labs(title= "variance = 0.1")

tmp %>%
  filter(variance == 0.5) %>%
  ggplot(aes(x = Z, y = marginal, color = model)) +
  geom_line() +
  facet_wrap(use_location ~ use_scale) +
  labs(title= "variance = 0.5")

### Experiment (e) ####
# "true marginal", "marginal by SURE-PM (the one without covariates)", and "marginal by NPMLE"
e_marginals_thin %>% 
  filter(model_raw %in% c("truth", "misspec", "wellspec", "surels", "NPMLE")) %>%
  ggplot(aes(x=Z, y = marginal)) +
  geom_line(aes(color=model, linetype=model, alpha=(model!="Bayes risk")), size=0.8) +
  facet_wrap( ~ variance, scales = "free", labeller=label_parsed) +
  scale_color_manual(values=NPMLE_misspec_truth, name="", breaks=c("NPMLE", "SURE-PM", "SURE-THING", "SURE-LS", "Bayes risk")) +
  theme(legend.position="bottom", text=element_text(size=13),
        legend.key.size = unit(3,"line")) +
  ylab(TeX(r"($f_G(z \, | \, sigma_i^2)$)")) +
  xlab(TeX("z")) +
  scale_linetype_manual(values=c(2, 4, 3, 5, 1), name="", breaks=c("NPMLE", "SURE-PM", "SURE-THING", "SURE-LS", "Bayes risk")) +
  guides(alpha="none") +
  scale_alpha_manual(values=c(1, 1))

ggsave("xie_plots_with_all/e_marginal_all_small.png", height=6, width=10)

e_marginals_thin %>% 
  filter(model_raw %in% c("truth", "misspec", "surels", "NPMLE")) %>%
  ggplot(aes(x=Z, y = marginal)) +
  geom_line(aes(color=model,  alpha=(model!="Bayes risk")), size=0.8) +
  facet_wrap( ~ variance, scales = "free", labeller=label_parsed) +
  scale_color_manual(values=NPMLE_misspec_truth, name="", breaks=c("NPMLE", "SURE-PM", "SURE-LS", "Bayes risk")) +
  theme(legend.position="bottom", text=element_text(size=13),
        legend.key.size = unit(3,"line")) +
  ylab(TeX(r"($f_G(z \, | \, sigma_i^2)$)")) +
  xlab(TeX("z")) +
  guides(alpha="none") +
  scale_alpha_manual(values=c(1, 1))

ggsave("xie_plots_with_all/e_marginal_misspec_surels_NPMLE_solid_small.png", height=6, width=10)

wellspec_true_colors = dark2_palette[c(8, 4, 3)]
wellspec_true_colors[1] = "#CCCCCC"
wellspec_true_colors = setNames(wellspec_true_colors, c("Bayes risk", "SURE-THING", "SURE-LS"))

e_marginals_thin %>% 
  filter(model_raw %in% c("truth", "wellspec", "surels")) %>%
  ggplot(aes(x=Z, y = marginal)) +
  geom_line(aes(color=model), size=0.8) +
  facet_wrap( ~ variance, scales = "free", labeller=label_parsed) +
  scale_color_manual(values=wellspec_true_colors, name="", breaks = c("SURE-THING", "SURE-LS", "Bayes risk")) +
  theme(legend.position="bottom", text=element_text(size=13))  +
  ylab(TeX(r"($f_G(z \, | \, sigma_i^2)$)")) +
  xlab(TeX("z")) 

ggsave("xie_plots_with_all/e_marginal_wellspec_surels_small.png", height=6, width=10)


