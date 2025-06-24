setwd("~/Documents/Github/empirical-bayes/results")
library(tidyverse)
library(ggh4x) # facetted_pos_scales
library(ggpubr) # arrange multiple figures together
library(latex2exp)
library(RColorBrewer)
library(lemon)
library(scales)
dark2_palette <- brewer.pal(8,"Dark2")
theme_set(theme_bw())


## In-sample MSE ##### 

df_names = list.files("xie_losses")
df_names = df_names[str_detect(df_names, "location_scale_comparison")]

df = NULL
for (df_name in df_names[-1]){ # ignore location_scale_comparison 0 
  
  if (is.null(df)){
    df = read.csv(paste("xie_losses", df_name, sep="/"))
  } else {
    df = df %>% bind_rows(read.csv(paste("xie_losses", df_name, sep="/")))
  }
  
}

df = df %>% 
  select(-X) %>% 
  pivot_longer(5:14, names_to = "model", values_to = "value") %>% 
  separate(model, c("objective", "model"))

df$use_location = as.logical(df$use_location)
df$use_scale = as.logical(df$use_scale)

df_mean = df %>%
  group_by(n, use_location, use_scale, experiment, data, objective, model) %>%
  summarize(mean = mean(value, na.rm=T),
            count = n(),
            se = sd(value, na.rm=T)/sqrt(count), .groups="drop")

view(df_mean)

#### Plot: !use location, use scale ####

m_sim = unique(df_mean$count)

bayes_risk = data.frame("experiment" = c("c", "d", "e", "f"),
                        "yintercept" = c(0, 0, 0.15, 0),
                        "model" = rep("Ground Truth", 4)) %>%
  mutate(experiment = factor(experiment))


df_mean_plot = df_mean %>%
  filter(data=="train", objective=="MSE",
         !use_location, use_scale, experiment != "d5",
         model != "NPMLEinit") %>%
  mutate(model = case_when(model == "misspec" ~ "SURE-PM",
                           model == "wellspec" ~ "SURE-THING",
                           model == "thetaG" ~ "SURE-grandmean",
                           TRUE ~ model),
         model = factor(model, levels=c("NPMLE", "SURE-grandmean", "SURE-PM", "SURE-THING", "Ground Truth"))) %>%
  mutate(experiment = factor(experiment),
         experiment = factor(experiment, levels=c("c", "d", "e", "f"))) 

ground_truth_df = df_mean_plot %>% 
  group_by(n, experiment) %>%
  summarize(count = n()) %>% 
  left_join(bayes_risk, by = "experiment") %>%
  mutate(model = factor(model, levels=c("NPMLE", "SURE-grandmean", "SURE-PM", "SURE-THING", "Ground Truth")),
         experiment = factor(experiment)) %>%
  rename(mean = yintercept)


levels(df_mean_plot$experiment) = c(c = TeX(r"(Uniform$$ prior )"),
                                    d = TeX(r"(Inv-$\chi^2$ prior)"),
                                    e = TeX(r"(Bimodal $\mu_i$)"),
                                    f = TeX(r"(Uniform$$ likelihood)"))

levels(ground_truth_df$experiment) = c(c = TeX(r"(Uniform$$ prior )"),
                                    d = TeX(r"(Inv-$\chi^2$ prior)"),
                                    e = TeX(r"(Bimodal $\mu_i$)"),
                                    f = TeX(r"(Uniform$$ likelihood)"))

levels(bayes_risk$experiment) = c(c = TeX(r"(Uniform$$ prior )"),
                                       d = TeX(r"(Inv-$\chi^2$ prior)"),
                                       e = TeX(r"(Bimodal $\mu_i$)"),
                                       f = TeX(r"(Uniform$$ likelihood)"))


df_mean_plot_line = df_mean_plot %>%
  select(n, experiment, count, mean, model,) %>%
  bind_rows(ground_truth_df)



ggplot(df_mean_plot, aes(x = factor(n), color=model, shape=model)) +
  geom_point(aes(y = mean), size = 2.5) +
  geom_line(data= df_mean_plot_line, aes(y = mean, group = model,  linetype=model)) +
  geom_errorbar(aes(ymin = mean-1.96*se, 
                    ymax = mean+1.96*se, 
                    group=model), alpha=0.5, show.legend=F) +
  facet_wrap( ~ experiment, scales="free_y", labeller=label_parsed) +
  # labs(title="In-sample MSE, scale only") +
  theme(axis.text.x = element_text(angle=45, hjust=1),
        legend.position="bottom",
        text=element_text(size=16),
        legend.key.size = unit(2,"line")) +
  scale_color_manual(values=dark2_palette[c(1:4,8)], name="") +
  scale_linetype_manual(values=c(2, 3, 4, 1, 1), name="") +
  scale_shape_manual(values=c(15, 16, 17, 18, 1), name="") +
  ylab("In-sample MSE") +
  xlab("n")

ggsave("ICSDS/median_scale_MSE_small.png", height=8.5, width=9)

ggplot(df_mean_plot, aes(x = factor(n), color=model,shape=model)) +
  geom_point(aes(y = mean), size = 3) +
  geom_line(data= df_mean_plot_line, aes(y = mean, group = model,  linetype=model), size=0.8) +
  geom_errorbar(aes(ymin = mean-1.96*se, 
                    ymax = mean+1.96*se, 
                    group=model), alpha=0.5, show.legend=F) +
  facet_wrap( ~ experiment, scales="free_y", labeller=label_parsed) +
  # labs(title="In-sample MSE, scale only") +
  theme(axis.text.x = element_text(angle=45, hjust=1),
        legend.position="bottom",
        text=element_text(size=24),
        legend.key.size = unit(3,"line")) +
  scale_color_manual(values=dark2_palette[c(1:4,8)], name="") +
  scale_linetype_manual(values=c(2, 3, 4, 1, 1), name="") +
  scale_shape_manual(values=c(15, 16, 17, 18, 1), name="") +
  ylab("In-sample MSE") +
  xlab("n")

ggsave("ICSDS/median_scale_MSE_large.png", height=12, width=13)


# bimodal only
df_mean_plot %>%
  filter(experiment == "\"Bimodal \" * mu[i]") %>%
  ggplot( aes(x = factor(n), color=model,shape=model)) +
  geom_point(aes(y = mean), size = 5) +
  geom_line(data= (df_mean_plot_line %>% filter(experiment == "\"Bimodal \" * mu[i]")), 
            aes(y = mean, group = model,  linetype=model), size=1.5) +
  geom_errorbar(aes(ymin = mean-1.96*se, 
                    ymax = mean+1.96*se, 
                    group=model), alpha=1, show.legend=F) +
  # facet_wrap( ~ experiment, scales="free_y", labeller=label_parsed) +
  # labs(title="In-sample MSE, scale only") +
  theme(axis.text.x = element_text(angle=45, hjust=1),
        legend.position="bottom",
        text=element_text(size=24),
        legend.key.size = unit(4,"line")) +
  scale_color_manual(values= c(dark2_palette[1:4], "#2E2E2E"), name="") +
  scale_linetype_manual(values=c(2, 3, 4, 1, 1), name="") +
  scale_shape_manual(values=c(15, 16, 17, 18, 1), name="") +
  ylab("In-sample MSE") +
  xlab("n")

ggsave("ICSDS/e_MSE_large.png", height=12, width=14)


# Read path ####

seed_str = "60050100"
read_path = paste("xie_checks_", seed_str, "/", sep="")

# Shrinkage / posterior means ####

xie_shrinkage = read.csv(paste(read_path, "xie_shrinkage_location_scale.csv", sep="")) %>% 
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

# "truth" "SURE-PM" and NPMLE

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

xie_shrinkage_e_thin %>%
  filter(model_raw %in% c("truth", "misspec_median_scale", "NPMLE")) %>%
  ggplot(aes(x = Z, y = posterior_mean, color = model, linetype=model)) +
  geom_line(size=1.5) +
  facet_wrap(~ variance, scales = "free_x", labeller=label_parsed) +
  # geom_hline(data = data.frame(variance = c(0.1, 0.5),
  #                              theta = c(2, 0)), 
  #            aes(yintercept = theta)) +
  scale_color_manual(values = NPMLE_misspec_truth, name="", breaks=c("NPMLE", "SURE-PM", "Ground truth")) +
  scale_linetype_manual(values=c(2, 4, 1), name="", breaks=c("NPMLE", "SURE-PM", "Ground truth")) +
  xlab(TeX("$z$")) +
  ylab("Posterior mean") +
  theme(legend.key.size = unit(3,"line"),
        text=element_text(size=24),
        axis.title.y=element_text(size=18))

ggsave("ICSDS/shrinkage_e_misspec_NPMLE_median_scale.png", width=10, height=6)

full_palette = c(dark2_palette[c(1:4)], "#CCCCCC")
full_palette = setNames(full_palette, c("NPMLE", "SURE-grandmean", "SURE-PM", "SURE-THING", "Ground truth"))

xie_shrinkage_e_thin %>% 
  ggplot(aes(x = Z, y = posterior_mean, color = model, linetype=model)) +
  geom_line(size=0.8) +
  facet_wrap(~ variance, scales = "free_x", labeller=label_parsed) +
  scale_color_manual(values = full_palette, name="", breaks = c("NPMLE", "SURE-grandmean", "SURE-PM", "SURE-THING", "Ground truth")) +
  scale_linetype_manual(values=c(2, 3, 4, 1, 1), name = "", breaks=c("NPMLE", "SURE-grandmean", "SURE-PM", "SURE-THING", "Ground truth"))+
  xlab(TeX("$z$")) +
  ylab("Posterior mean") +
  theme(legend.key.size = unit(2,"line"))


ggsave("ICSDS/shrinkage_e_all_median_scale.png", width=6.5, height=4)

xie_shrinkage_e_thin %>% 
  ggplot(aes(x = Z, y = posterior_mean, color = model)) +
  geom_line(size=0.8) +
  facet_wrap(~ variance, scales = "free_x", labeller=label_parsed) +
  scale_color_manual(values = full_palette, name="", breaks = c("NPMLE", "SURE-grandmean", "SURE-PM", "SURE-THING", "Ground truth")) +
  scale_linetype_manual(values=c(2, 3, 4, 1, 1), name = "", breaks=c("NPMLE", "SURE-grandmean", "SURE-PM", "SURE-THING", "Ground truth"))+
  xlab(TeX("$z$")) +
  ylab("Posterior mean") +
  theme(legend.key.size = unit(2,"line"))


ggsave("ICSDS/shrinkage_e_all_median_scale_solid.png", width=6.5, height=4)




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
  pivot_longer(6:11, names_to = "model", values_to = "marginal") %>% 
  filter(model != 'NPMLEinit') %>%
  arrange(model) %>%
  mutate(experiment = factor(experiment)) %>%
  mutate(model_raw = model,
         model = case_when(model == "misspec" ~ "SURE-PM",
                           model == "wellspec" ~ "SURE-THING",
                           model == "thetaG" ~ "SURE-grandmean",
                           model == "truth" ~ "Ground truth",
                           TRUE ~ model),
         # switch the order of models so that ground truth is in the back of the plot
         model = factor(model, levels = c("Ground truth", "SURE-THING", "SURE-PM", "SURE-grandmean",  "NPMLE" )),
         experiment = factor(experiment, levels=c("c", "d", "f", "e")))

marginals_med_scale_thin = marginals_med_scale[seq(1, dim(marginals_med_scale)[1], 5), ]

e_marginals_thin = marginals_med_scale_thin %>% 
  filter(experiment == "e") %>%
  mutate(variance = factor(variance))


NPMLE_misspec_truth = dark2_palette[c(8, 3, 1)]
NPMLE_misspec_truth[1] = "#9E9E9E"
NPMLE_misspec_truth = setNames(NPMLE_misspec_truth, rev(c("NPMLE", "SURE-PM", "Ground truth")))



levels(e_marginals_thin$variance) = c(TeX(r"($\sigma_i^2 = 0.1$)"),
                                      TeX(r"($\sigma_i^2 = 0.5$)"))


tmp = marginals_df %>% 
  filter(experiment == "e") %>% 
  pivot_longer(6:11, names_to = "model", values_to = "marginal") %>% 
  filter(model %in% c("truth", "wellspec", "misspec", "NPMLE"))

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

ggsave("ICSDS/e_marginal_misspec_NPMLE_small.png", height=6, width=10)

e_marginals_thin %>% 
  filter(model_raw %in% c("truth", "misspec", "NPMLE")) %>%
  ggplot(aes(x=Z, y = marginal)) +
  geom_line(aes(color=model,  alpha=(model!="Ground truth")), size=0.8) +
  facet_wrap( ~ variance, scales = "free", labeller=label_parsed) +
  scale_color_manual(values=NPMLE_misspec_truth, name="", breaks=c("NPMLE", "SURE-PM", "Ground truth")) +
  theme(legend.position="bottom", text=element_text(size=13),
        legend.key.size = unit(3,"line")) +
  ylab(TeX(r"($f_G(z \, | \, sigma_i^2)$)")) +
  xlab(TeX("z")) +
  guides(alpha="none") +
  scale_alpha_manual(values=c(1, 1))

ggsave("ICSDS/e_marginal_misspec_NPMLE_solid_small.png", height=6, width=10)

wellspec_true_colors = dark2_palette[c(8, 4)]
wellspec_true_colors[1] = "#CCCCCC"
wellspec_true_colors = setNames(wellspec_true_colors, c("Ground truth", "SURE-THING"))



e_marginals_thin %>% 
  filter(model_raw %in% c("truth", "wellspec")) %>%
  ggplot(aes(x=Z, y = marginal)) +
  geom_line(aes(color=model), size=0.8) +
  facet_wrap( ~ variance, scales = "free", labeller=label_parsed) +
  scale_color_manual(values=wellspec_true_colors, name="", breaks = c("SURE-THING", "Ground truth")) +
  theme(legend.position="bottom", text=element_text(size=13))  +
  ylab(TeX(r"($f_G(z \, | \, sigma_i^2)$)")) +
  xlab(TeX("z")) 

ggsave("ICSDS/e_marginal_wellspec_small.png", height=6, width=10)

### all experiments ####
# TODO: Fix the color ordering on this

# levels(marginals_med_scale_thin$experiment) = c(c = TeX(r"(Uniform$$ prior )"),
#                                     d = TeX(r"(Inv-$\chi^2$ prior)"),
#                                     f = TeX(r"(Uniform$$ likelihood)"),
#                                     e = TeX(r"(Bimodal $\mu_i$)"))
# 
# marginals_med_scale_thin %>% 
#   ggplot(aes(x=Z, y = marginal)) +
#   geom_line(aes(color=model)) +
#   facet_wrap(experiment ~ variance, nrow=4, scales = "free", labeller=label_parsed) +
#   scale_color_manual(values=dark2_palette[c(1:4, 8)]) +
#   theme(legend.position="bottom", text=element_text(size=13)) +
#   ylab(TeX(r"($f_G(z \, | \, sigma_i^2)$)")) +
#   xlab(TeX("z"))
# 
# ggsave("ICSDS/marginal_all_large.png", height=12, width=10)

# Marginals (e) with posterior means ####

### Find quantiles of marginal ####

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


### 0.1 ####

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

ggsave("ICSDS/marginal_posterior_1.png", width = 4, height=5)

 ### 0.5 ####

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

ggsave("ICSDS/marginal_posterior_5.png", width = 4, height=5)

## Combined 0.1 and 0.5 #### 

combined_marginal_and_shrinkage = marginal_and_shrinkage_01 %>%
  mutate(variance = 'low') %>% 
  bind_rows((marginal_and_shrinkage_05 %>% mutate(variance = 'high'))) %>%
  mutate(variance = factor(variance),
         name = factor(name),
         plotting_trick = ifelse(name == "A", "`Posterior mean`", "Marginal"),
         plotting_trick = factor(plotting_trick, levels = c("`Posterior mean`", "Marginal")))

levels(combined_marginal_and_shrinkage$variance) = c(low = TeX(r"($\sigma_i^2 = 0.1$)"),
                                                     high = TeX(r"($\sigma_i^2 = 0.5$)"))


combined_ribbon_marginal = ribbon_marginal_05 %>% mutate(variance = 'high') %>%
  bind_rows((ribbon_marginal_01 %>% mutate(variance = 'low'))) %>%
  mutate(variance = factor(variance),
         plotting_trick = "Marginal",
         plotting_trick = factor(plotting_trick, levels = c("`Posterior mean`", "Marginal")))

levels(combined_ribbon_marginal$variance) = c(low = TeX(r"($\sigma_i^2 = 0.1$)"),
                                                     high = TeX(r"($\sigma_i^2 = 0.5$)"))



ggplot() +
  geom_area(data=combined_ribbon_marginal, aes(x=Z, y=y), fill="#9EEEEE", alpha=0.4) +
  geom_line(data=combined_marginal_and_shrinkage %>% 
              mutate(keep_posterior_mean_to_truncate_plot = 
                       !( variance == "sigma[i]^{\n    2\n} * {\n    phantom() == phantom()\n} * 0.5" & name == "A" & value < 1)) %>% 
              filter(keep_posterior_mean_to_truncate_plot, Z >= -3, Z <= 3),
            aes(x=Z, y = value, color=model, linetype=model, alpha=alpha_95), size=0.8) +
  facet_rep_grid(rows= vars(plotting_trick), cols = vars(variance), scales="free",
                 labeller = label_parsed,
                 switch="y",
                 repeat.tick.labels = T) +
  # facet_wrap(plotting_trick ~ variance, scales="free_y",
  #                labeller = label_parsed) +
  scale_color_manual(values=NPMLE_misspec_truth, name="", breaks=c("NPMLE", "SURE-PM", "Ground truth")) +
  theme(legend.position="bottom", text=element_text(size=13),
        legend.key.size = unit(3,"line"),
        strip.background = element_blank(),
        strip.placement = 'outside') +
  ylab(TeX(r"($f_G(z \, | \, sigma_i^2)$)")) +
  xlab(TeX("z")) +
  scale_linetype_manual(values=c(2, 4, 1), name="", breaks=c("NPMLE", "SURE-PM", "Ground truth")) +
  guides(alpha="none") +
  scale_alpha_manual(values=c(1, .25, .25)) +
  ylab("") 

ggsave("ICSDS/combined_marginal_shrinkage_truncate_axes.png", width = 7, height=5)


ggplot() +
  geom_area(data=combined_ribbon_marginal, aes(x=Z, y=y), fill="#9EEEEE", alpha=0.4) +
  geom_line(data=combined_marginal_and_shrinkage,
            aes(x=Z, y = value, color=model, linetype=model, alpha=alpha_95), size=0.8) +
  facet_rep_grid(rows= vars(plotting_trick), cols = vars(variance), scales="free",
                 labeller = label_parsed,
                 switch="y",
                 repeat.tick.labels = T) +
  scale_color_manual(values=NPMLE_misspec_truth, name="", breaks=c("NPMLE", "SURE-PM", "Ground truth")) +
  theme(legend.position="bottom", text=element_text(size=13),
        legend.key.size = unit(3,"line"),
        strip.background = element_blank(),
        strip.placement = 'outside') +
  ylab(TeX(r"($f_G(z \, | \, sigma_i^2)$)")) +
  xlab(TeX("z")) +
  scale_linetype_manual(values=c(2, 4, 1), name="", breaks=c("NPMLE", "SURE-PM", "Ground truth")) +
  guides(alpha="none") +
  scale_alpha_manual(values=c(1, .25, .25)) +
  ylab("") 

ggsave("ICSDS/combined_marginal_shrinkage_whole_axes.png", width = 7, height=5)



# Expanded ####

exp_marginals_df = read.csv(paste(read_path, "location_scale_marginals_expanded.csv", sep="")) %>% 
  select(-X) 
# 6400 n x 4 location scale x 5 experments x 3 variances
exp_marginals_df$use_scale = as.logical(marginals_df$use_scale)
exp_marginals_df$use_location = as.logical(marginals_df$use_location)

exp_marginals_med_scale = exp_marginals_df %>% 
  filter(use_scale, !use_location) %>%
  filter(experiment != "d5") %>% 
  pivot_longer(6:11, names_to = "model", values_to = "marginal") %>% 
  filter(model != 'NPMLEinit') %>%
  arrange(model) %>%
  mutate(experiment = factor(experiment)) %>%
  mutate(model_raw = model,
         model = case_when(model == "misspec" ~ "SURE-PM",
                           model == "wellspec" ~ "SURE-THING",
                           model == "thetaG" ~ "SURE-grandmean",
                           model == "truth" ~ "Ground truth",
                           TRUE ~ model),
         # switch the order of models so that ground truth is in the back of the plot
         model = factor(model, levels = c("Ground truth", "SURE-THING", "SURE-PM", "SURE-grandmean",  "NPMLE" )),
         experiment = factor(experiment, levels=c("c", "d", "f", "e")))



exp_marginals_med_scale_thin = exp_marginals_med_scale[seq(1, dim(exp_marginals_med_scale)[1], 5), ]

exp_e_marginals_thin = exp_marginals_med_scale_thin %>% 
  filter(experiment == "e") %>%
  mutate(variance = factor(variance))


levels(exp_e_marginals_thin$variance) = c(TeX(r"($\sigma_i^2 = 0.1$)"),
                                      TeX(r"($\sigma_i^2 = 0.5$)"))


### Experiment (e) ####
# "true marginal", "marginal by SURE-PM (the one without covariates)", and "marginal by NPMLE"
exp_e_marginals_thin %>% 
  filter(model_raw %in% c("truth", "misspec")) %>%
  ggplot(aes(x=Z, y = marginal)) +
  geom_line(aes(color=model, linetype=model, alpha=(model!="Ground truth")), size=0.8) +
  facet_wrap( ~ variance, scales = "free", labeller=label_parsed) +
  scale_color_manual(values=NPMLE_misspec_truth[2:3], name="", breaks=c("SURE-PM", "Ground truth")) +
  theme(legend.position="bottom", text=element_text(size=13),
        legend.key.size = unit(3,"line")) +
  ylab(TeX(r"($f_G(z \, | \, sigma_i^2)$)")) +
  xlab(TeX("z")) +
  scale_linetype_manual(values=c( 4, 1), name="", breaks=c("SURE-PM", "Ground truth")) +
  guides(alpha="none") +
  scale_alpha_manual(values=c(1, 1))




# Priors (e) ####

priors_raw =  read.csv(paste(read_path, "location_scale_priors.csv", sep="")) %>%
  select(-X)

priors = priors_raw %>%
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
  unnest(cols=c(theta, pi))


prior_e = priors %>%
  filter(experiment == "e", model != "NPMLEinit") %>%
  mutate(model = case_when(model == "misspec" ~ "SURE-PM",
                           model == "wellspec" ~ "SURE-THING",
                           model == "thetaG" ~ "SURE-grandmean",
                           TRUE ~ model), 
         model = factor(model, levels = c("NPMLE","SURE-PM",  "Ground Truth")),
         plotting_trick = factor(model,levels = c("NPMLE","SURE-PM",  "Ground Truth")))

levels(ground_truth_e$model) = c(low = TeX(r"(T $\sigma^2_i = 0.1$)"),
                                 high = TeX(r"(T $\sigma^2_i = 0.5$)"))

theta_truth_01 = seq(min(prior_e$theta), max(prior_e$theta), by = 0.01)
theta_truth_05 = seq(min(prior_e$theta), max(prior_e$theta), by = 0.01)
pi_truth_01 = dnorm(theta_truth_01, mean=2, sd=sqrt(0.1)) # low variance
pi_truth_05 = dnorm(theta_truth_05, mean=0, sd=sqrt(0.5)) # high variance
ground_truth_e =  data.frame('experiment' = rep("e", 2*length(theta_truth_01)),
                             'model' = factor(c(rep("low", length(theta_truth_01)),
                                        rep("high", length(theta_truth_01)))),
                             theta=c(theta_truth_01, theta_truth_05),
                             pi=c(pi_truth_01, pi_truth_05))

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
        legend.key.size = unit(3,"line")) +
  xlab(TeX("$theta_j$")) +
  ylab(TeX("$pi_j")) +
  scale_x_continuous(limit = c(-5, 5))

# In order to have the sigma_i legend show up at the end,
# first run the plot code without levels(ground_truth$...)
# so that the legend shows low and high.

# And then fix the labels, and somehow it'll show up second.

ggsave("ICSDS/e_prior.png", height=6, width=10)



prior_e %>% filter(model == "SURE-PM") %>%
  group_by(theta > 5) %>% 
  summarize(count = n(),
            prior_mass_above_5 = sum(pi),
            max_theta = max(theta),
            min_theta = min(theta))


# Empirical marginals ####

empirical_marginals_df = read.csv(paste(read_path, "location_scale_empirical_marginals.csv", sep="")) %>% 
  select(-X)
# 6400 n x 4 location scale x 5 experiments 
empirical_marginals_df$use_scale = as.logical(empirical_marginals_df$use_scale)
empirical_marginals_df$use_location = as.logical(empirical_marginals_df$use_location)

empirical_marginals_med_scale = empirical_marginals_df %>% 
  filter(use_scale, !use_location) %>%
  filter(experiment != "d5") %>% 
  pivot_longer(5:10, names_to = "model", values_to = "marginal") %>% 
  filter(model != 'NPMLEinit') %>%
  arrange(model) %>%
  mutate(experiment = factor(experiment)) %>%
  mutate(model_raw = model,
         model = case_when(model == "misspec" ~ "SURE-PM",
                           model == "wellspec" ~ "SURE-THING",
                           model == "thetaG" ~ "SURE-grandmean",
                           model == "truth" ~ "Ground truth",
                           TRUE ~ model),
         model = factor(model, levels = rev(c("NPMLE", "SURE-grandmean", "SURE-PM", "SURE-THING", "Ground truth"))),
         experiment = factor(experiment, levels=c("c", "d", "f", "e"))) 


empirical_marginals_med_scale_thin = empirical_marginals_med_scale[seq(1, dim(empirical_marginals_med_scale)[1], 10), ]

e_empirical_marginals_thin = empirical_marginals_med_scale_thin %>% 
  filter(experiment == "e") 

# levels(marginals_med_scale_thin$experiment) = c(c = TeX(r"(Uniform$$ prior )"),
#                                     d = TeX(r"(Inv-$\chi^2$ prior)"),
#                                     f = TeX(r"(Uniform$$ likelihood)"),
#                                     e = TeX(r"(Bimodal $\mu_i$)"))



# "true marginal", "marginal by SURE-PM (the one without covariates)", and "marginal by NPMLE"
e_empirical_marginals_thin %>% 
  filter(model_raw %in% c("truth", "misspec", "NPMLE")) %>%
  ggplot(aes(x=Z, y = marginal/6400)) +
  geom_line(aes(color=model, linetype=model), size=1.5) +
  scale_color_manual(values=NPMLE_misspec_truth, name="", breaks=c("NPMLE", "SURE-PM", "Ground truth")) +
  scale_linetype_manual(values=c(2, 4, 1), name ="", breaks=c("NPMLE", "SURE-PM", "Ground truth")) +
  theme(legend.position="bottom", text=element_text(size=24),
        legend.key.size = unit(4, "line"),
        axis.title.y=element_text(size=16)) +
  ylab(TeX(r"($\frac{1}{n} \sum_i \, f_G(z \, | \, sigma_i^2)$)")) +
  xlab(TeX("z"))

ggsave("ICSDS/e_empirical_marginal_misspec_NPMLE_small.png", height=6, width=10)


e_empirical_marginals_thin %>% 
  filter(model_raw %in% c("truth", "misspec", "NPMLE")) %>%
  ggplot(aes(x=Z, y = marginal/6400)) +
  geom_line(aes(color=model), size=0.8) +
  scale_color_manual(values=NPMLE_misspec_truth, name="", breaks=c("NPMLE", "SURE-PM", "Ground truth")) +
  theme(legend.position="bottom", text=element_text(size=13)) +
  ylab(TeX(r"($\frac{1}{n} \sum_i \, f_G(z \, | \, sigma_i^2)$)")) +
  xlab(TeX("z"))

ggsave("ICSDS/e_empirical_marginal_misspec_NPMLE_solid_small.png", height=6, width=10)


e_empirical_marginals_thin %>% 
  filter(model_raw %in% c("truth", "wellspec")) %>%
  ggplot(aes(x=Z, y = marginal/6400)) +
  geom_line(aes(color=model), size=0.8) +
  scale_color_manual(values=wellspec_true_colors, name="", breaks=c("SURE-THING", "Ground truth")) +
  theme(legend.position="bottom", text=element_text(size=13)) +
  ylab(TeX(r"($\frac{1}{n} \sum_i \, f_G(z \, | \, sigma_i^2)$)")) +
  xlab(TeX("z"))

ggsave("ICSDS/e_empirical_marginal_wellspec_small.png", height=6, width=10)

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

