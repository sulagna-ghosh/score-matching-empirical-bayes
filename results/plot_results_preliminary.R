setwd("~/Documents/Github/empirical-bayes/results")
library(tidyverse)
library(ggh4x) # facetted_pos_scales
library(ggpubr) # arrange multiple figures together
# theme_set(theme_pubr())


# Covariates ####

covariates = read.csv("covariates.csv") %>%
  select(name, sigma, rhos, MLE_MSE, EB_MSE, NPMLE_MSE) %>%
  pivot_longer(4:6, names_to="estimator", values_to="MSE")

covariates %>%
  filter(str_detect(name, "homoskedastic")) %>%
  ggplot() +
  geom_boxplot(aes(x = estimator, y = MSE), outliers = FALSE) +
  facet_wrap(sigma ~ name, scales="free_y") + 
  facetted_pos_scales(
    y = rep(list(
      scale_y_continuous(limits = c(0.5, 1.4)),
      scale_y_continuous(limits = c(1, 10))
    ), each = 2)
  ) +
  ggtitle("Well-specified, homoeskedastic")

covariates %>%
  filter(str_detect(name, "heteroskedastic")) %>%
  ggplot(aes(x = estimator, y = MSE)) +
  geom_boxplot(outliers = FALSE) +
  facet_wrap(sigma ~ name, scales="free_y") + 
  facetted_pos_scales(
    y = rep(list(
      scale_y_continuous(limits = c(0.5, 1.4)),
      scale_y_continuous(limits = c(1, 6.5))
    ), each = 2)
  ) +
  ggtitle("Well-specified, heteroskedastic")

covariates %>%
  ggplot(aes(x = rhos, y = MSE, color = estimator)) +
  geom_point(alpha=0.4) 


## Misspecified ##### 

dropsigma = read.csv("covariates_dropsigma.csv")  %>%
  select(name, sigma, rhos, MLE_MSE, EB_MSE, NPMLE_MSE) %>%
  pivot_longer(4:6, names_to="estimator", values_to="MSE")

dropsigma %>%
  ggplot(aes(x = estimator, y = MSE)) +
  geom_boxplot(outliers = FALSE) +
  facet_wrap(sigma ~ name, scales="free_y") + 
  facetted_pos_scales(
    y = rep(list(
      scale_y_continuous(limits = c(0.7, 1.5)),
      scale_y_continuous(limits = c(3, 6.5))
    ), each = 2)
  ) +
  ggtitle("Misspecified - extra sigma")



# no_covariates ####

## No covariates: MSE and SURE ####

### Normal theta case (Homoscedastic) ####



nocovariates_normal  = NULL

for (file in list.files("no_covariates", pattern="normal_homo_*")){
  
  file_name = paste0("no_covariates/", file)
  
  if (is.null(nocovariates_binary)){
    
    nocovariates_normal = read.csv(file_name) 
    
  } else {
    
    nocovariates_normal = nocovariates_normal %>%
      bind_rows(read.csv(file_name) ) 
    
  }
  
}


# select 50 from each level
nocovariates_normal_var_0.1 = nocovariates_normal %>%
  filter(sigma_theta < 0.317) %>% 
  head(50)

nocovariates_normal_var_1 = nocovariates_normal %>%
  filter(sigma_theta > 0.317 & sigma_theta < 1.1) %>% 
  head(50)

nocovariates_normal_var_2.2 = nocovariates_normal %>%
  filter(sigma_theta > 2.2) %>% 
  head(50)

nocovariates_normal_50 = bind_rows(nocovariates_normal_var_0.1,
                                   nocovariates_normal_var_1,
                                   nocovariates_normal_var_2.2) %>%
  select(sigma_theta, SURE_2step_MSE, SURE_both_MSE, SURE_theta_MSE, SURE_pi_MSE, SURE_sparse_MSE, NPMLE_MSE, SURE_NPMLEinit_MSE, BAYES_MSE) %>%
  pivot_longer(2:9, names_to="estimator", values_to="MSE")
nocovariates_normal_50$estimator_name <- substr(nocovariates_normal_50$estimator, 1, nchar(nocovariates_normal_50$estimator) - 4)
head(nocovariates_normal_50) 


nocovariates_normal_50 %>%
  filter(estimator_name %in% c("SURE_both", "NPMLE", "BAYES") ) %>%
  group_by(sigma_theta, estimator_name) %>%
  summarize(mean_sum_squared_errors = mean(MSE),
            count = n()) %>% view()




# 
# nocovariates_normal = read.csv("nocovariates_normal_homo.csv") %>%
#   select(sigma_theta, SURE_2step_MSE, SURE_both_MSE, SURE_theta_MSE, SURE_pi_MSE, SURE_sparse_MSE, NPMLE_MSE, SURE_NPMLEinit_MSE, BAYES_MSE) %>%
#   pivot_longer(2:9, names_to="estimator", values_to="MSE") 
# 
# nocovariates_normal$estimator_name <- substr(nocovariates_normal$estimator, 1, nchar(nocovariates_normal$estimator) - 4)
# 
# head(nocovariates_normal)
# 
# nocovariates_normal %>% ggplot() +
#   geom_boxplot(aes(x = estimator_name, y = MSE), outliers = FALSE) +
#   facet_wrap(~ sigma_theta, scales = "free_y") +
#   theme(axis.text.x = element_text(angle=45, vjust = 1, hjust=1)) + 
#   ggtitle("Homoskedastic - Normal theta - MSE")
# 
# nocovariates_normal = read.csv("nocovariates_normal_homo.csv") %>%
#   select(sigma_theta, SURE_2step_MSE, SURE_both_MSE, SURE_theta_MSE, SURE_pi_MSE, SURE_sparse_MSE, NPMLE_MSE, SURE_NPMLEinit_MSE, BAYES_MSE)
# nocovariates_normal_mean <- aggregate(. ~ sigma_theta, data = nocovariates_normal, FUN = mean)
# nocovariates_normal_mean
# write.csv(nocovariates_normal_mean,"nocovariates_normal_homo_mean.csv",row.names = F)
# 
# nocovariates_normal = read.csv("nocovariates_normal_homo.csv") %>%
#   select(sigma_theta, SURE_NPMLEinit_OBJ_LOSS, SURE_OBJ_LOSS, SURE_NPMLEinit_LOSS, SURE_both_LOSS) %>%
#   pivot_longer(2:5, names_to="estimator", values_to="LOSS") 
# nocovariates_normal$estimator_name <- substr(nocovariates_normal$estimator, 1, nchar(nocovariates_normal$estimator) - 5)
# nocovariates_normal %>% ggplot() +
#   geom_boxplot(aes(x = estimator_name, y = LOSS), outliers = FALSE) +
#   facet_wrap(~ sigma_theta, scales = "free_y") +
#   theme(axis.text.x = element_text(angle=45, vjust = 1, hjust=1)) + 
#   ggtitle("Homoskedastic - Normal theta - LOSS")


### Normal theta case (Heteroscedastic) ####


nocovariates_normal_hetero = read.csv("nocovariates_normal_hetero.csv") %>%
  select(sigma_theta, SURE_2step_MSE, SURE_both_MSE, SURE_theta_MSE, SURE_pi_MSE, SURE_sparse_MSE, NPMLE_MSE, SURE_NPMLEinit_MSE, BAYES_MSE) %>%
  pivot_longer(2:9, names_to="estimator", values_to="MSE") 

nocovariates_normal_hetero$estimator_name <- substr(nocovariates_normal_hetero$estimator, 1, nchar(nocovariates_normal_hetero$estimator) - 4)

head(nocovariates_normal_hetero)

nocovariates_normal_hetero %>% ggplot() +
  geom_boxplot(aes(x = estimator_name, y = MSE), outliers = FALSE) +
  facet_wrap(~ sigma_theta, scales = "free_y") +
  theme(axis.text.x = element_text(angle=45, vjust = 1, hjust=1)) + 
  ggtitle("Heteroskedastic - Normal theta - MSE") 

nocovariates_normal_hetero = read.csv("nocovariates_normal_hetero.csv") %>%
  select(sigma_theta, SURE_2step_MSE, SURE_both_MSE, SURE_theta_MSE, SURE_pi_MSE, SURE_sparse_MSE, NPMLE_MSE, SURE_NPMLEinit_MSE, BAYES_MSE)
nocovariates_normal_hetero_mean <- aggregate(. ~ sigma_theta, data = nocovariates_normal_hetero, FUN = mean)
nocovariates_normal_hetero_mean
write.csv(nocovariates_normal_hetero_mean,"nocovariates_normal_hetero_mean.csv",row.names = F)

nocovariates_normal_hetero = read.csv("nocovariates_normal_hetero.csv") %>%
  select(sigma_theta, SURE_NPMLEinit_OBJ_LOSS, SURE_OBJ_LOSS, SURE_NPMLEinit_LOSS, SURE_both_LOSS) %>%
  pivot_longer(2:5, names_to="estimator", values_to="LOSS") 
nocovariates_normal_hetero$estimator_name <- substr(nocovariates_normal_hetero$estimator, 1, nchar(nocovariates_normal_hetero$estimator) - 5)
nocovariates_normal_hetero %>% ggplot() +
  geom_boxplot(aes(x = estimator_name, y = LOSS), outliers = FALSE) +
  facet_wrap(~ sigma_theta, scales = "free_y") +
  theme(axis.text.x = element_text(angle=45, vjust = 1, hjust=1)) + 
  ggtitle("Heteroskedastic - Normal theta - LOSS") 

# With only SURE both and NPMLE results (with varying n)

nocovariates_normal_hetero = read.csv("nocovariates_normal_hetero_varyn.csv") %>%
  select(sigma_theta, n, SURE_MSE, NPMLE_MSE, SURE_NPMLEinit_MSE, BAYES_MSE) %>%
  pivot_longer(3:6, names_to="estimator", values_to="MSE") 

nocovariates_normal_hetero$estimator_name <- substr(nocovariates_normal_hetero$estimator, 1, nchar(nocovariates_normal_hetero$estimator) - 4)
head(nocovariates_normal_hetero)

nocovariates_normal_hetero %>% ggplot() +
  geom_boxplot(aes(x = estimator_name, y = MSE), outliers = FALSE) +
  facet_wrap(sigma_theta ~ n, scales = "free_y") +
  theme(axis.text.x = element_text(angle=45, vjust = 1, hjust=1)) + 
  ggtitle("Heteroskedastic - Normal theta - Vary n - MSE") 

nocovariates_normal_hetero = read.csv("nocovariates_normal_hetero_varyn.csv") %>%
  select(sigma_theta, n, SURE_LOSS, SURE_NPMLEinit_OBJ_LOSS, SURE_NPMLEinit_LOSS) %>%
  pivot_longer(3:5, names_to="estimator", values_to="LOSS") 

nocovariates_normal_hetero$estimator_name <- substr(nocovariates_normal_hetero$estimator, 1, nchar(nocovariates_normal_hetero$estimator) - 5)

nocovariates_normal_hetero %>% ggplot() +
  geom_boxplot(aes(x = estimator_name, y = LOSS), outliers = FALSE) +
  facet_wrap(sigma_theta ~ n, scales = "free_y") +
  theme(axis.text.x = element_text(angle=45, vjust = 1, hjust=1)) + 
  ggtitle("Heteroskedastic - Normal theta - Vary n - LOSS") 

### Binary theta case (Homoscedastic) ####

nocovariates_binary  = NULL

for (file in list.files("no_covariates", pattern="binary_homo_*")){
  
  file_name = paste0("no_covariates/", file)
  
  if (is.null(nocovariates_binary)){
    
    nocovariates_binary = read.csv(file_name) 
    
  } else {
    
    nocovariates_binary = nocovariates_binary %>%
      bind_rows(read.csv(file_name) ) 
    
  }
  
}

remove_idx_for_50_sim = c(1, 2, 3, 4, 13, 14, 15, 16, 25, 26, 27, 28)
nocovariates_binary = nocovariates_binary[-remove_idx_for_50_sim,] %>%
  group_by(k, mu) %>%
  select(mu, k, SURE_2step_MSE, SURE_both_MSE, SURE_theta_MSE, SURE_pi_MSE, SURE_sparse_MSE, NPMLE_MSE, SURE_NPMLEinit_MSE, BAYES_MSE) %>%
  pivot_longer(3:10, names_to="estimator", values_to="MSE")
nocovariates_binary$estimator_name <- substr(nocovariates_binary$estimator, 1, nchar(nocovariates_binary$estimator) - 4)
head(nocovariates_binary) 

nocovariates_binary %>% 
  filter(estimator_name %in% c("SURE_both")) %>%
  ggplot() +
  geom_boxplot(aes(x = estimator_name, y = MSE*1000), outliers = FALSE) +
  facet_wrap(mu ~ k, scales = "free_y", nrow=4) +
  theme(axis.text.x = element_text(angle=45, vjust = 1, hjust=1)) + 
  ggtitle("Homoskedastic - Binary theta - MSE")


nocovariates_binary %>%
  filter(estimator_name %in% c("SURE_both", "NPMLE", "BAYES") ) %>%
  group_by(mu, k, estimator_name) %>%
  summarize(mean_sum_squared_errors = mean(MSE),
            count = n()) %>% view()


# 
# nocovariates_binary = read.csv("nocovariates_binary_homo.csv") %>%
#   select(mu, k, SURE_2step_MSE, SURE_both_MSE, SURE_theta_MSE, SURE_pi_MSE, SURE_sparse_MSE, NPMLE_MSE, SURE_NPMLEinit_MSE, BAYES_MSE)
# nocovariates_binary_mean <- aggregate(. ~ mu + k, data = nocovariates_binary, FUN = mean)
# nocovariates_binary_mean
# write.csv(nocovariates_binary_mean,"nocovariates_binary_homo_mean.csv",row.names = F)
# 
# nocovariates_binary = read.csv("nocovariates_binary_homo.csv") %>%
#   select(mu, k, SURE_NPMLEinit_OBJ_LOSS, SURE_OBJ_LOSS, SURE_NPMLEinit_LOSS, SURE_both_LOSS) %>%
#   pivot_longer(3:6, names_to="estimator", values_to="LOSS") 
# nocovariates_binary$estimator_name <- substr(nocovariates_binary$estimator, 1, nchar(nocovariates_binary$estimator) - 5)
# nocovariates_binary %>% ggplot() +
#   geom_boxplot(aes(x = estimator_name, y = LOSS), outliers = FALSE) +
#   facet_wrap(mu ~ k, scales = "free_y") +
#   theme(axis.text.x = element_text(angle=45, vjust = 1, hjust=1)) + 
#   ggtitle("Homoskedastic - Binary theta - LOSS") 


### Binary theta case (Heteroscedastic) ####

nocovariates_binary_hetero = read.csv("nocovariates_binary_hetero.csv") %>%
  select(mu, k, SURE_2step_MSE, SURE_both_MSE, SURE_theta_MSE, SURE_pi_MSE, SURE_sparse_MSE, NPMLE_MSE, SURE_NPMLEinit_MSE, BAYES_MSE) %>%
  pivot_longer(3:10, names_to="estimator", values_to="MSE")
nocovariates_binary_hetero$estimator_name <- substr(nocovariates_binary_hetero$estimator, 1, nchar(nocovariates_binary_hetero$estimator) - 4)
head(nocovariates_binary_hetero) 

nocovariates_binary_hetero %>% ggplot() +
  geom_boxplot(aes(x = estimator_name, y = MSE), outliers = FALSE) +
  facet_wrap(mu ~ k, scales = "free_y") +
  theme(axis.text.x = element_text(angle=45, vjust = 1, hjust=1)) + 
  ggtitle("Heteroskedastic - Binary theta - MSE") 


nocovariates_binary_hetero = read.csv("nocovariates_binary_hetero.csv") %>%
  select(mu, k, SURE_2step_MSE, SURE_both_MSE, SURE_theta_MSE, SURE_pi_MSE, SURE_sparse_MSE, NPMLE_MSE, SURE_NPMLEinit_MSE, BAYES_MSE)
nocovariates_binary_hetero_mean <- aggregate(. ~ mu + k, data = nocovariates_binary_hetero, FUN = mean)
nocovariates_binary_hetero_mean
write.csv(nocovariates_binary_hetero_mean,"nocovariates_binary_hetero_mean.csv",row.names = F)

nocovariates_binary_hetero = read.csv("nocovariates_binary_hetero.csv") %>%
  select(mu, k, SURE_NPMLEinit_OBJ_LOSS, SURE_OBJ_LOSS, SURE_NPMLEinit_LOSS, SURE_both_LOSS) %>%
  pivot_longer(3:6, names_to="estimator", values_to="LOSS") 
nocovariates_binary_hetero$estimator_name <- substr(nocovariates_binary_hetero$estimator, 1, nchar(nocovariates_binary_hetero$estimator) - 5)
nocovariates_binary_hetero %>% ggplot() +
  geom_boxplot(aes(x = estimator_name, y = LOSS), outliers = FALSE) +
  facet_wrap(mu ~ k, scales = "free_y") +
  theme(axis.text.x = element_text(angle=45, vjust = 1, hjust=1)) + 
  ggtitle("Heteroskedastic - Binary theta - LOSS") 

# Score visualization (Binary)

# Homoskedastic - fitted

score_nocovariates_binary = read.csv("score_nocovariates_binary_homo_test.csv") %>% 
  select(n, mu, k, Z, NPMLE, SURE)  %>% 
  pivot_longer(5:6, names_to = "Estimator", values_to = "Score")
head(score_nocovariates_binary) 

ggplot(score_nocovariates_binary, aes(x=Z, y=Score, group=Estimator)) +
  geom_line(aes(color=Estimator)) + 
  facet_wrap(mu ~ k, scales = "free_y") + 
  ggtitle("Homoskedastic - Binary theta - Fitted score") 

# Homoskedastic - estimated

score_nocovariates_binary = read.csv("score_nocovariates_binary_homo_test.csv") %>% 
  select(n, mu, k, Z_grid, NPMLE_grid, SURE_grid)  %>% 
  pivot_longer(5:6, names_to = "Estimator", values_to = "Score")
head(score_nocovariates_binary) 

ggplot(score_nocovariates_binary, aes(x=Z_grid, y=Score, group=Estimator)) +
  geom_line(aes(color=Estimator)) + 
  facet_wrap(mu ~ k, scales = "free_y") + 
  ggtitle("Homoskedastic - Binary theta - Estimated score") 

# Heteroskedastic - fitted

score_nocovariates_binary = read.csv("score_nocovariates_binary_hetero_test.csv") %>% 
  select(n, mu, k, Z, NPMLE, SURE)  %>% 
  pivot_longer(5:6, names_to = "Estimator", values_to = "Score")
head(score_nocovariates_binary) 

ggplot(score_nocovariates_binary, aes(x=Z, y=Score, group=Estimator)) +
  geom_line(aes(color=Estimator)) + 
  facet_wrap(mu ~ k, scales = "free_y") + 
  ggtitle("Heteroskedastic - Binary theta - Fitted score") 

# Heteroskedastic - estimated

score_nocovariates_binary = read.csv("score_nocovariates_binary_hetero_test.csv") %>% 
  select(n, mu, k, Z_grid, NPMLE_grid, SURE_grid)  %>% 
  pivot_longer(5:6, names_to = "Estimator", values_to = "Score")
head(score_nocovariates_binary) 

ggplot(score_nocovariates_binary, aes(x=Z_grid, y=Score, group=Estimator)) +
  geom_line(aes(color=Estimator)) + 
  facet_wrap(mu ~ k, scales = "free_y") + 
  ggtitle("Heteroskedastic - Binary theta - Estimated score") 



## B comparison - Test #####


### Normal theta case ####

nocovariates_normal_B = read.csv("nocovariates_normal_B.csv") %>%
  select(sigma_theta, SURE_pi_B100_MSE, SURE_pi_B500_MSE) %>%
  pivot_longer(2:3, names_to="estimator", values_to="MSE") 
nocovariates_normal_B$estimator_name <- substr(nocovariates_normal_B$estimator, 9, nchar(nocovariates_normal_B$estimator) - 4)
head(nocovariates_normal_B) 

nocovariates_normal_B %>%
  ggplot() +
  geom_boxplot(aes(x = estimator_name, y = MSE), outliers = FALSE) +
  facet_wrap(~ sigma_theta, scales="free_y") + 
  ggtitle("Homoskedastic - Normal theta - B comparison") 

nocovariates_normal_B = read.csv("nocovariates_normal_B.csv") %>%
  select(sigma_theta, SURE_pi_B100_MSE, SURE_pi_B500_MSE) 

nocovariates_normal_B_mean <- aggregate(. ~ sigma_theta, data = nocovariates_normal_B, FUN = mean)
nocovariates_normal_B_mean
write.csv(nocovariates_normal_B_mean,"nocovariates_normal_B_mean.csv",row.names = F)


### Binary theta case ####

nocovariates_binary_B = read.csv("nocovariates_binary_B.csv") %>%
  select(mu, k, SURE_pi_B100_MSE, SURE_pi_B500_MSE) %>%
  pivot_longer(3:4, names_to="estimator", values_to="MSE") 
nocovariates_binary_B$estimator_name <- substr(nocovariates_binary_B$estimator, 9, nchar(nocovariates_binary_B$estimator) - 4)
head(nocovariates_binary_B) 

nocovariates_binary_B %>%
  ggplot() +
  geom_boxplot(aes(x = estimator_name, y = MSE), outliers = FALSE) +
  facet_wrap(mu ~ k, scales="free_y") + 
  ggtitle("Homoskedastic - Binary theta - B comparison") 

nocovariates_binary_B = read.csv("nocovariates_binary_B.csv") %>%
  select(mu, k, SURE_pi_B100_MSE, SURE_pi_B500_MSE) 
nocovariates_binary_B_mean <- aggregate(. ~ mu + k, data = nocovariates_binary_B, FUN = mean)
nocovariates_binary_B_mean
write.csv(nocovariates_binary_B_mean,"nocovariates_binary_B_mean.csv",row.names = F)



# scores ####

### Normal Homoskedastic - fitted ####

score_nocovariates_normal = read.csv("score_nocovariates_normal_homo.csv") %>% 
  select(n, sigma_theta, Z, NPMLE, SURE)  %>% 
  pivot_longer(4:5, names_to = "Estimator", values_to = "Score")
head(score_nocovariates_normal) 

ggplot(score_nocovariates_normal, aes(x=Z, y=Score, group=Estimator)) +
  geom_line(aes(color=Estimator)) + 
  facet_wrap(n ~ sigma_theta, scales = "free_y") + 
  ggtitle("Homoskedastic - Normal theta - Fitted score") 

score_nocovariates_normal_subset = score_nocovariates_normal[which(score_nocovariates_normal['sigma_theta'] == 5), ]
head(score_nocovariates_normal_subset) 

ggplot(score_nocovariates_normal_subset, aes(x=Z, y=Score, group=Estimator)) +
  geom_line(aes(color=Estimator)) + 
  facet_wrap( ~ n, scales = "free_y") + 
  ggtitle("Homoskedastic - Normal(10, 25) - Fitted score") 


### Homoskedastic - estimated ####

score_nocovariates_normal = read.csv("score_nocovariates_normal_homo.csv") %>% 
  select(n, sigma_theta, Z_grid, NPMLE_grid, SURE_grid, TRUTH_grid)  %>% 
  pivot_longer(4:6, names_to = "Estimator", values_to = "Score")
head(score_nocovariates_normal) 

ggplot(score_nocovariates_normal, aes(x=Z_grid, y=Score, group=Estimator)) +
  geom_line(aes(color=Estimator)) + 
  facet_wrap(n ~ sigma_theta, scales = "free_y") + 
  ggtitle("Homoskedastic - Normal theta - Estimated score") 

score_nocovariates_normal_subset = score_nocovariates_normal[which(score_nocovariates_normal['sigma_theta'] == 5), ]
head(score_nocovariates_normal_subset) 

ggplot(score_nocovariates_normal_subset, aes(x=Z_grid, y=Score, group=Estimator)) +
  geom_line(aes(color=Estimator)) + 
  facet_wrap( ~ n, scales = "free_y") + 
  ggtitle("Homoskedastic - Normal(10, 25) - Estimated score") 

### Heteroskedastic - fitted ####

score_nocovariates_normal = read.csv("score_nocovariates_normal_hetero.csv") %>% 
  select(n, sigma_i, sigma_theta, Z, NPMLE, SURE)  %>% 
  pivot_longer(5:6, names_to = "Estimator", values_to = "Score")
head(score_nocovariates_normal) 

# ggplot(score_nocovariates_normal, aes(x=Z, y=Score, group=Estimator)) +
# geom_line(aes(color=Estimator)) + 
# facet_wrap(n ~ sigma_theta ~ sigma_i, scales = "free_y") + 
# ggtitle("Heteroskedastic - Normal theta - Fitted score") 

score_nocovariates_normal_subset = score_nocovariates_normal[which(score_nocovariates_normal['sigma_theta'] == 5), ]
head(score_nocovariates_normal_subset) 

ggplot(score_nocovariates_normal_subset, aes(x=Z, y=Score, group=Estimator)) +
  geom_line(aes(color=Estimator)) + 
  facet_wrap(n ~ sigma_i, scales = "free_y") + 
  ggtitle("Heteroskedastic - Normal(10, 25) - Fitted score")



### Heteroskedastic - estimated ####

score_nocovariates_normal = read.csv("score_nocovariates_normal_hetero.csv") %>% 
  select(n, sigma_i, sigma_theta, Z_grid, NPMLE_grid, SURE_grid, TRUTH_grid)  %>% 
  pivot_longer(5:7, names_to = "Estimator", values_to = "Score")
head(score_nocovariates_normal) 

# ggplot(score_nocovariates_normal, aes(x=Z_grid, y=Score, group=Estimator)) +
# geom_line(aes(color=Estimator)) + 
# facet_wrap(~ sigma_theta, scales = "free_y") + 
# ggtitle("Heteroskedastic - Normal theta - Estimated score") 

score_nocovariates_normal_subset = score_nocovariates_normal[which(score_nocovariates_normal['sigma_theta'] == 5), ]
head(score_nocovariates_normal_subset) 

ggplot(score_nocovariates_normal_subset, aes(x=Z_grid, y=Score, group=Estimator)) +
  geom_line(aes(color=Estimator, lty = Estimator), size=0.4) + 
  facet_wrap(n ~ sigma_i, scales = "free_y", nrow=3) + 
  ggtitle("Heteroskedastic - Normal(10, 25) - Estimated score") +
  theme(legend.position = "bottom",
        strip.text.x = element_text(size = 6),
        legend.text.position = "bottom",
        text = element_text(size=8))

ggsave("no_covariates/figures_normal_theta/nocovariates_normaltheta_hetero_EstimatedScore_var25_varynsigma.png")

score_nocovariates_normal_subsubset = score_nocovariates_normal_subset[which(score_nocovariates_normal_subset['n'] == 1000), ]
score_nocovariates_normal_subsubset = score_nocovariates_normal_subsubset[which(score_nocovariates_normal_subsubset['sigma_i'] == 1), ]
head(score_nocovariates_normal_subsubset) 

ggplot(score_nocovariates_normal_subsubset, aes(x=Z_grid, y=Score, group=Estimator)) +
  geom_line(aes(color=Estimator)) + 
  ggtitle("Heteroskedastic - Normal(10, 25) - Estimated score - n = 1000") +
  theme(legend.position = "bottom",
        strip.text.x = element_text(size = 6),
        legend.text.position = "bottom",
        text = element_text(size=8))

ggsave("no_covariates/figures_normal_theta/nocovariates_normaltheta_hetero_EstimatedScore_var25_varysigma_n1000.png")

score_nocovariates_normal_subset = score_nocovariates_normal[which(score_nocovariates_normal['sigma_theta'] == 5), ]
score_nocovariates_normal_subsubset = score_nocovariates_normal_subset[which(score_nocovariates_normal_subset['sigma_i'] == 1), ]
head(score_nocovariates_normal_subsubset) 

ggplot(score_nocovariates_normal_subsubset, aes(x=Z_grid, y=Score, group=Estimator)) +
  geom_line(aes(color=Estimator)) + 
  # geom_point(size = 0.05) + 
  facet_wrap( ~ n, scales = "free_y", nrow=3) + 
  ggtitle("Heteroskedastic - Normal(10, 25) - Estimated score") +
  theme(legend.position = "bottom",
        strip.text.x = element_text(size = 6),
        legend.text.position = "bottom",
        text = element_text(size=8))

ggsave("no_covariates/figures_normal_theta/nocovariates_normaltheta_hetero_EstimatedScore_var25_varyn.png")

# Estimated theta

theta_hat_nocovariates_normal = read.csv("score_nocovariates_normal_hetero.csv") %>% 
  select(n, sigma_i, sigma_theta, Z_grid, theta_hat_NPMLE_grid, theta_hat_SURE_grid, theta_hat_TRUTH_grid)  %>% 
  pivot_longer(5:7, names_to = "Estimator", values_to = "theta_hat")
head(theta_hat_nocovariates_normal) 

theta_hat_nocovariates_normal_subset = theta_hat_nocovariates_normal[which(theta_hat_nocovariates_normal['sigma_theta'] == 5), ]
head(theta_hat_nocovariates_normal_subset) 

ggplot(theta_hat_nocovariates_normal_subset, aes(x=Z_grid, y=theta_hat, group=Estimator)) +
  geom_line(aes(color=Estimator, lty = Estimator), size=0.4) + 
  facet_wrap(n ~ sigma_i, scales = "free_y", nrow=3) + 
  ggtitle("Heteroskedastic - Normal(10, 25) - Estimated theta") +
  theme(legend.position = "bottom",
        strip.text.x = element_text(size = 6),
        legend.text.position = "bottom",
        text = element_text(size=8))

ggsave("no_covariates/figures_normal_theta/nocovariates_normaltheta_hetero_Estimatedtheta_var25_varynsigma.png")

theta_hat_nocovariates_normal_subsubset = theta_hat_nocovariates_normal_subset[which(theta_hat_nocovariates_normal_subset['n'] == 1000), ]
theta_hat_nocovariates_normal_subsubset = theta_hat_nocovariates_normal_subsubset[which(theta_hat_nocovariates_normal_subsubset['sigma_i'] == 1), ]
head(theta_hat_nocovariates_normal_subsubset) 

ggplot(theta_hat_nocovariates_normal_subsubset, aes(x=Z_grid, y=theta_hat, group=Estimator)) +
  geom_line(aes(color=Estimator)) + 
  ggtitle("Heteroskedastic - Normal(10, 25) - Estimated theta - n = 1000") +
  theme(legend.position = "bottom",
        strip.text.x = element_text(size = 6),
        legend.text.position = "bottom",
        text = element_text(size=8))

ggsave("no_covariates/figures_normal_theta/nocovariates_normaltheta_hetero_Estimatedtheta_var25_varysigma_n1000.png")

theta_hat_nocovariates_normal_subset = theta_hat_nocovariates_normal[which(theta_hat_nocovariates_normal['sigma_theta'] == 5), ]
theta_hat_nocovariates_normal_subsubset = theta_hat_nocovariates_normal_subset[which(theta_hat_nocovariates_normal_subset['sigma_i'] == 1), ]
head(theta_hat_nocovariates_normal_subsubset) 

ggplot(theta_hat_nocovariates_normal_subsubset, aes(x=Z_grid, y=theta_hat, group=Estimator)) +
  geom_line(aes(color=Estimator)) + 
  # geom_point(size = 0.05) + 
  facet_wrap( ~ n, scales = "free_y", nrow=3) + 
  ggtitle("Heteroskedastic - Normal(10, 25) - Estimated theta") +
  theme(legend.position = "bottom",
        strip.text.x = element_text(size = 6),
        legend.text.position = "bottom",
        text = element_text(size=8))

ggsave("no_covariates/figures_normal_theta/nocovariates_normaltheta_hetero_Estimatedtheta_var25_varyn.png")

### Xie scores #### 

score_xie = read.csv("score_xie.csv") %>% select(-X.1)

xie_score_clean = score_xie %>%
  select(-Z) %>%
  mutate(`Truth?` = case_when(experiment != "e" ~ -1/sigma**2 * (Z_grid - sigma**2),
                              experiment == "e" & (round(sigma**2, 2) == 0.5) ~ -1/(2*sigma**2) * (Z_grid ),
                              experiment == "e" & (sigma**2 == 0.1) ~ -1/(2*sigma**2) * (Z_grid -2))) %>%
  pivot_longer(c(1:4, 10), names_to = "model", values_to = "score") %>% 
  mutate(model = case_when(model == "score_NPMLE" ~ model,
                           model == "Truth?" ~ model,
                           TRUE ~ paste("EB_", model, sep=""))) 

xie_score_clean %>% 
  filter(experiment == "c") %>%
  ggplot(aes(x = Z_grid, y = score, color = model, lty=model)) +
  geom_line() +
  facet_wrap(round(sigma**2, 3) ~ n, nrow = 3, scales = "free_y") +
  theme(legend.position = "bottom",
        strip.text.x = element_text(size = 6),
        legend.text.position = "bottom",
        text = element_text(size=8)) +
  labs(title="c") 

ggsave("scores/figures/xie_scores_c.png")

xie_score_clean %>% 
  filter(experiment == "d") %>%
  ggplot(aes(x = Z_grid, y = score, color = model, lty = model)) +
  geom_line() +
  facet_wrap(round(sigma**2, 3) ~ n, nrow = 3, scales = "free_y") +
  theme(legend.position = "bottom",
        strip.text.x = element_text(size = 6),
        legend.text.position = "bottom",
        text = element_text(size=8)) + 
  labs(title="d")

ggsave("scores/figures/xie_scores_d.png")

xie_score_clean %>% 
  filter(experiment == "e") %>%
  ggplot(aes(x = Z_grid, y = score, color = model, lty = model)) +
  geom_line() +
  facet_wrap(round(sigma**2, 3) ~ n, nrow = 2, scales = "free_y") +
  theme(legend.position = "bottom",
        strip.text.x = element_text(size = 6),
        legend.text.position = "bottom",
        text = element_text(size=8)) + 
  labs(title="e")

ggsave("scores/figures/xie_scores_e.png")

xie_score_clean %>% 
  filter(experiment == "f") %>%
  ggplot(aes(x = Z_grid, y = score, color = model, lty = model)) +
  geom_line() +
  facet_wrap(round(sigma**2, 3) ~ n, nrow = 3, scales = "free_y") +
  theme(legend.position = "bottom",
        strip.text.x = element_text(size = 6),
        legend.text.position = "bottom",
        text = element_text(size=8)) + 
  labs(title="f")

ggsave("scores/figures/xie_scores_f.png")



# xie_checks ####

## shrinkage ####

xie_shrinkage = read.csv("xie_shrinkage.csv") %>% select(-X)

xie_shrinkage %>% 
  filter(experiment == "c") %>%
  pivot_longer(c(1:5), names_to = "model", values_to = "theta_hat") %>%
  ggplot(aes(x = Z, y = theta_hat, color = model)) +
  geom_line() +
  facet_wrap(~ variance, scales = "free") +
  geom_hline(aes(yintercept = variance)) +
  labs(title="Experiment (c)")

ggsave("xie_checks/figures_shrinkages/shrinkage_c.png")

xie_shrinkage %>% 
  filter(experiment == "d") %>%
  pivot_longer(c(1:5), names_to = "model", values_to = "theta_hat") %>%
  ggplot(aes(x = Z, y = theta_hat, color = model)) +
  geom_line() +
  facet_wrap(~ variance, scales = "free") +
  geom_hline(aes(yintercept = variance)) +
  labs(title="Experiment (d)")

ggsave("xie_checks/figures_shrinkages/shrinkage_d.png")

xie_shrinkage %>% 
  filter(experiment == "d5") %>%
  pivot_longer(c(1:5), names_to = "model", values_to = "theta_hat") %>%
  ggplot(aes(x = Z, y = theta_hat, color = model)) +
  geom_line() +
  facet_wrap(~ variance, scales = "free") +
  geom_hline(aes(yintercept = variance)) +
  labs(title="Experiment (d5)")

ggsave("xie_checks/figures_shrinkages/shrinkage_d5.png")

xie_shrinkage %>% 
  filter(experiment == "e") %>%
  pivot_longer(c(1:6), names_to = "model", values_to = "theta_hat") %>%
  ggplot(aes(x = Z, y = theta_hat, color = model)) +
  geom_line() +
  facet_wrap(~ variance, scales = "free") +
  geom_hline(data = data.frame(variance = c(0.1, 0.5),
                               theta = c(2, 0)), 
             aes(yintercept = theta)) +
  labs(title="Experiment (e)")

ggsave("xie_checks/figures_shrinkages/shrinkage_e.png")

xie_shrinkage %>% 
  filter(experiment == "f") %>%
  pivot_longer(c(1:5), names_to = "model", values_to = "theta_hat") %>%
  ggplot(aes(x = Z, y = theta_hat, color = model)) +
  geom_line() +
  facet_wrap(~ variance, scales = "free") +
  geom_hline(aes(yintercept = variance)) +
  labs(title="Experiment (f)")

ggsave("xie_checks/figures_shrinkages/shrinkage_f.png")



# xie_losses ####

mse_sure_256 = read.csv("xie_losses/mse_sure_256_256.csv")

mse_sure = read.csv("xie_losses/mse_sure_1.csv")

m_replicates = mse_sure %>%
  group_by(n, experiment, data) %>%
  summarize(count = n()) %>%
  pull(count) %>% max()

## Train #####
# Expect training loss to go to 0 for wellspecified

mse_sure %>% 
  filter(data=="train") %>% 
  select(n, experiment, SURE_wellspec) %>%
  ggplot(aes(x = factor(n), y = SURE_wellspec, group=n)) +
  geom_boxplot() +
  facet_wrap(~ experiment, nrow=2, scales="free")  +
  geom_hline(yintercept=0, color="red", linetype="dashed") +
  labs(title="Training error of wellspec with outliers")

ggsave("xie_losses/figures_xie_losses/8_8/wellspec_outliers_small_n.png", height=9, width=9)

mse_sure %>% 
  filter(data=="train") %>% 
  select(n, experiment, SURE_wellspec) %>%
  ggplot(aes(x = factor(n), y = SURE_wellspec, group=n)) +
  geom_boxplot(outliers=F) +
  facet_wrap(~ experiment, nrow=2, scales="free")  +
  geom_hline(yintercept=0, color="red", linetype="dashed") +
  labs(title="Training error of wellspec excluding outliers")

ggsave("xie_losses/figures_xie_losses/8_8/wellspec_no_outliers_small_n.png", height=9, width=9)

### Mean training error ####

mean_se_train_sure = mse_sure %>% 
  filter(data=="train") %>% 
  select(n, experiment, starts_with("SURE")) %>%
  pivot_longer(3:7, names_to = "model", values_to="SURE") %>%
  mutate(model=substr(model, 6, nchar(model))) %>%
  group_by(n, experiment, model) %>%
  summarize(mean_SURE = mean(SURE), 
            se_SURE = sd(SURE)/sqrt(m_replicates), .groups="drop") 

mean_se_train_sure %>%
  # filter(model != "NPMLE") %>%
  ggplot(aes(x = factor(n))) +
  geom_point(aes(y = mean_SURE, color = model),) +
  geom_line(aes(y = mean_SURE, group = model, color=model)) +
  geom_errorbar(aes(ymin = mean_SURE-1.96*se_SURE, 
                    ymax = mean_SURE+1.96*se_SURE, 
                    group=model, color=model), alpha=0.5) +
  facet_wrap(~ experiment, scales="free_y") +
  labs(title="Training error",
       subtitle=paste(m_replicates, "replicates")) +
  theme(axis.text.x = element_text(angle=45, hjust=1),
        legend.position="bottom") +
  ylab("SURE") +
  xlab("log(n)")

ggsave("xie_losses/figures_xie_losses/8_8/training_error.png", height=9, width=9)

### In-sample MSE ####

mean_se_train_MSE = mse_sure %>% 
  filter(data=="train") %>% 
  select(n, experiment, starts_with("MSE")) %>%
  pivot_longer(3:7, names_to = "model", values_to="MSE") %>%
  mutate(model=substr(model, 5, nchar(model))) %>%
  group_by(n, experiment, model) %>%
  summarize(mean_MSE = mean(MSE), 
            se_MSE = sd(MSE)/sqrt(m_replicates), .groups="drop") 

mean_se_train_MSE %>%
  filter(model != "NPMLEinit") %>%
  ggplot(aes(x = factor(n))) +
  geom_point(aes(y = mean_MSE, color = model),) +
  geom_line(aes(y = mean_MSE, group = model, color=model)) +
  geom_errorbar(aes(ymin = mean_MSE-1.96*se_MSE, 
                    ymax = mean_MSE+1.96*se_MSE, 
                    group=model, color=model), alpha=0.5) +
  facet_wrap(~ experiment, scales="free_y") +
  labs(title="In-sample MSE",
       subtitle=paste(m_replicates, "replicates")) +
  theme(axis.text.x = element_text(angle=45, hjust=1),
        legend.position="bottom") +
  ylab("MSE") +
  xlab("log(n)")



ggsave("xie_losses/figures_xie_losses/8_8/in_sample_MSE.png", height=9, width=9)

mean_se_train_MSE %>%
  ggplot(aes(x = factor(n))) +
  geom_point(aes(y = mean_MSE, color = model),) +
  geom_line(aes(y = mean_MSE, group = model, color=model)) +
  geom_errorbar(aes(ymin = mean_MSE-1.96*se_MSE, 
                    ymax = mean_MSE+1.96*se_MSE, 
                    group=model, color=model), alpha=0.5) +
  facet_wrap(~ experiment, scales="free_y") +
  labs(title="In-sample MSE",
       subtitle=paste(m_replicates, "replicates")) +
  theme(axis.text.x = element_text(angle=45, hjust=1),
        legend.position="bottom") +
  ylab("MSE") +
  xlab("log(n)")

### In-sample MSE_NPMLEinit - MSE_misspec ####

mse_sure %>%
  filter(data=="train", !(experiment == "f" & MSE_NPMLEinit-MSE_misspec > .01)) %>%
  ggplot() +
  facet_wrap(~experiment, nrow=3, scales="free") +
  geom_boxplot(aes(x=factor(n), group=factor(n), y=(MSE_NPMLEinit-MSE_misspec)/MSE_misspec)) +
  ylab("In-sample (MSE_NPMLEinit - MSE_misspec)/MSE_misspec") +
  xlab("log(n)") +
  labs(title="How much does initialization help?") 

ggsave("xie_losses/figures_xie_losses/8_8/in_sample_MSE_NPMLEinit_misspec.png", height=10, width=9)

## Test MSE (8,8) #####

mean_se_test_MSE = mse_sure %>% 
  filter(data=="test") %>% 
  select(n, experiment, starts_with("MSE")) %>%
  pivot_longer(3:7, names_to = "model", values_to="MSE") %>%
  mutate(model=substr(model, 5, nchar(model))) %>%
  group_by(n, experiment, model) %>%
  summarize(mean_MSE = mean(MSE), 
            se_MSE = sd(MSE)/sqrt(m_replicates), .groups="drop") 

mean_se_test_MSE %>%
  #filter(model != "NPMLE") %>%
  ggplot(aes(x = factor(n))) +
  geom_point(aes(y = mean_MSE, color = model),) +
  geom_line(aes(y = mean_MSE, group = model, color=model)) +
  geom_errorbar(aes(ymin = mean_MSE-1.96*se_MSE, 
                    ymax = mean_MSE+1.96*se_MSE, 
                    group=model, color=model), alpha=0.5) +
  facet_wrap(~ experiment, scales="free_y") +
  labs(title="Out-sample MSE",
       subtitle=paste(m_replicates, "replicates")) +
  theme(axis.text.x = element_text(angle=45, hjust=1),
        legend.position="bottom") +
  ylab("MSE") +
  xlab("log(n)")

ggsave("xie_losses/figures_xie_losses/8_8/out_sample_MSE.png", height=9, width=9)


## Test (256, 256) #####

### All MSE #####
for (experiment_str in c("c", "d", "d5", "e", "f")){
  
  plot = mse_sure_test %>%
    select(n, experiment, starts_with("MSE")) %>%
    pivot_longer(3:7, values_to = "MSE") %>%
    mutate(estimator = substr(name, 5, nchar(name))) %>%
    filter(experiment == experiment_str, MSE != -100) %>%
    ggplot() +
    geom_boxplot(aes(x = estimator, y = MSE, group = estimator,
                     color = estimator)) +
    facet_wrap(~ n, nrow=2, scales = "free_y") +
    theme(axis.text.x = element_text(angle=45, vjust = 1, hjust=1),
          legend.position = "none",strip.text.x = element_text(size = 6),
          axis.text = element_text(size = 6)) +
    ylab("MSE (test)") +
    labs(title = paste("Experiment", experiment_str))
  
  path = "xie_losses/figures_xie_losses/mse_all/"
  filename = paste(path, experiment_str, "_test_boxplot.png", sep="")
  
  ggsave(filename, plot)
  
}

### All MSE means ####

mse_sure_test %>%
  select(n, experiment, starts_with("MSE")) %>%
  pivot_longer(3:7, values_to = "MSE") %>%
  filter(MSE != -100) %>%
  group_by(n, experiment, name) %>%
  summarize(`Mean MSE` = mean(MSE),
            sd_MSE = sd(MSE)) %>%
  mutate(estimator = substr(name, 5, nchar(name))) %>%
  ggplot(aes(x = n, y = `Mean MSE`, color=estimator)) +
  geom_point() +
  geom_line() +
  geom_errorbar(aes(ymin=`Mean MSE` - 1.96*sd_MSE, ymax=`Mean MSE` + 1.96*sd_MSE), width=.2,
                position=position_dodge(0.05)) +
  facet_wrap(~ experiment, scales="free_y")  + 
  ylab("Mean MSE (Test)")

filename = paste(path, "test_means.png", sep="")
ggsave()

### no covariates MSE, exclude failures #####
for (experiment_str in c("c", "d", "d5", "e", "f")){
  
  plot = mse_sure_test %>%
    select(n, experiment, starts_with("MSE")) %>%
    pivot_longer(3:7, values_to = "MSE") %>%
    mutate(estimator = substr(name, 5, nchar(name))) %>%
    filter(experiment == experiment_str) %>%
    filter(estimator != "wellspec", MSE != -100) %>%
    ggplot() +
    geom_boxplot(aes(x = estimator, y = MSE, group = estimator,
                     color = estimator)) +
    facet_wrap(~ n, nrow=2, scales = "free_y") +
    theme(axis.text.x = element_text(angle=45, vjust = 1, hjust=1),
          legend.position = "none",strip.text.x = element_text(size = 6),
          axis.text = element_text(size = 6)) +
    ylab("MSE (test)") +
    labs(title = paste("Experiment", experiment_str))
  
  path = "xie_losses/figures_xie_losses/mse_no_covariates/"
  filename = paste(path, experiment_str, "_test_boxplot.png", sep="")
  
  ggsave(filename, plot)
  
}

### No covariates MSE, means ####

mse_sure_test %>%
  select(n, experiment, starts_with("MSE")) %>%
  pivot_longer(3:7, values_to = "MSE") %>%
  filter(MSE != -100) %>%
  group_by(n, experiment, name) %>%
  summarize(`Mean MSE` = mean(MSE),
            sd_MSE = sd(MSE), .groups="drop") %>%
  mutate(estimator = substr(name, 5, nchar(name))) %>%
  filter(estimator != "wellspec", estimator != "NPMLE") %>%
  ggplot(aes(x = n, y = `Mean MSE`, group=estimator, color=estimator)) +
  geom_point(aes(shape=estimator)) + 
  geom_line() +
  geom_point(aes(y=`Mean MSE` - 1.96*sd_MSE), shape=3) +
  geom_point(aes(y=`Mean MSE` + 1.96*sd_MSE), shape=3) +
  facet_wrap(~ experiment, scales="free_y")  + 
  ylab("Mean MSE (Test)") +
  labs(subtitle="±1.96*sd marked as (+)")

filename = paste(path, "test_means.png", sep="")
ggsave(filename)


mse_sure_test %>%
  select(n, experiment, starts_with("MSE")) %>%
  pivot_longer(3:7, values_to = "MSE") %>%
  filter(MSE != -100) %>%
  group_by(n, experiment, name) %>%
  summarize(`Mean MSE` = mean(MSE),
            sd_MSE = sd(MSE)) %>%
  mutate(estimator = substr(name, 5, nchar(name))) %>%
  filter(estimator != "wellspec", estimator != "NPMLE") %>%
  ggplot(aes(x = n, y = `Mean MSE`, color=estimator)) +
  geom_point() +
  geom_line() +
  facet_wrap(~ experiment, scales="free_y")  + 
  ylab("Mean MSE (Test)")

filename = paste(path, "test_means_no_error_bar.png", sep="")
ggsave(filename)

mse_sure_test %>%
  select(n, experiment, starts_with("MSE")) %>%
  pivot_longer(3:7, values_to = "MSE") %>%
  filter(MSE != -100) %>%
  group_by(n, experiment, name) %>%
  summarize(`Mean MSE` = mean(MSE),
            sd_MSE = sd(MSE)) %>%
  mutate(estimator = substr(name, 5, nchar(name))) %>%
  filter(estimator != "wellspec") %>%
  ggplot(aes(x = n, y = `Mean MSE`, color=estimator)) +
  geom_point() +
  geom_line() +
  facet_wrap(~ experiment, scales="free_y")  + 
  ylab("Mean MSE (Test)")

filename = paste(path, "test_means_no_error_bar_NPMLE.png", sep="")
ggsave(filename)


### (d) only -- no covariates MSE, exclude failures #####

mse_sure_test %>%
  filter(MSE_NPMLEinit != -100, experiment == "d") %>%
  ggplot(aes(x = MSE_misspec - MSE_NPMLEinit)) +
  geom_histogram() +
  facet_wrap(~ n, scales="free")
ggsave("xie_losses/figures_xie_losses/mse_d/misspec_NPMLEinit_difference.png")
  


ggsave("figures/xie_test_MSE_means.png")

xie_test %>%
  select(n, experiment, starts_with("SURE")) %>%
  pivot_longer(3:7, values_to = "SURE") %>%
  filter(SURE != -100) %>%
  group_by(n, experiment, name) %>%
  summarize(mean_SURE = mean(SURE)) %>%
  mutate(estimator = substr(name, 6, nchar(name))) %>%
  ggplot(aes(x = n, y = mean_SURE, color=estimator)) +
  geom_point() +
  geom_line() +
  facet_wrap(~ experiment, scales="free_y")

ggsave("figures/xie_test_SURE_means.png")


## Location and scale ##### 

df_names = list.files("xie_losses/")
df_names = df_names[str_detect(df_names, "location_scale_comparison")]

df = NULL
for (df_name in df_names){
  
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


# when does mosek fail?
df %>% 
  filter(model=="NPMLE") %>%
  group_by(n, use_location, use_scale, experiment, data, objective, is.na(value)) %>%
  summarize(count = n(), .groups="drop") %>% view()

#### Plot by experiment

for (experiment_letter in c("c", "d", "d5", "e", "f")){
  
  if (experiment_letter == "e") {
    yintercept = 0.15
  } else {yintercept=0}
  
  plot = df_mean %>%
    filter(data=="train", objective=="MSE", experiment == experiment_letter)  %>%
    mutate(should_be_the_same = str_detect(model, "NPMLE$|thetaG")) %>%
    mutate(location = ifelse(use_location, "Learn location", "Use median(Z)"),
           scale = ifelse(use_scale, "Learn scale", "Use IQR(Z)")) %>%
    ggplot(aes(x = factor(n), alpha=!should_be_the_same)) +
    geom_point(aes(y = mean, color = model),) +
    geom_line(aes(y = mean, group = model, color=model)) +
    geom_errorbar(aes(ymin = mean-1.96*se, 
                      ymax = mean+1.96*se, 
                      group=model, color=model), alpha=0.5) +
    geom_hline(aes(yintercept=yintercept)) +
    facet_wrap(scale ~ location) +
    ylab("In-sample MSE") +
    guides(alpha="none") +
    labs(title=paste("Experiment (", experiment_letter, ")", sep="")) +
    theme(axis.text.x = element_text(angle=45, hjust=1),
          legend.position="bottom",
          text = element_text(size=12)) +
    xlab("log(n)")
         
  filename=paste("xie_losses/figures_xie_losses/full_", experiment_letter, "_MSE.png", sep="")
  ggsave(filename, plot)
}

for (experiment_letter in c("c", "d", "d5", "e", "f")){
  
  
  plot = df_mean %>%
    filter(data=="train", objective=="MSE", experiment == experiment_letter)  %>%
    filter( n > 400, model %in% c("misspec", "thetaG", "NPMLEinit")) %>%
    mutate(should_be_the_same = str_detect(model, "NPMLE$|thetaG")) %>%
    mutate(location = ifelse(use_location, "Learn location", "Use median(Z)"),
           scale = ifelse(use_scale, "Learn scale", "Use IQR(Z)")) %>%
    ggplot(aes(x = factor(n))) +
    geom_point(aes(y = mean, color = model),) +
    geom_line(aes(y = mean, group = model, color=model)) +
    geom_errorbar(aes(ymin = mean-1.96*se, 
                      ymax = mean+1.96*se, 
                      group=model, color=model), alpha=0.5) +
    facet_wrap(scale ~ location) +
    ylab("In-sample MSE") +
    guides(alpha="none") +
    labs(title=paste("Larger n: Experiment (", experiment_letter, ")", sep="")) +
    theme(axis.text.x = element_text(angle=45, hjust=1),
          legend.position="bottom",
          text = element_text(size=12)) +
    xlab("log(n)")
  
  filename=paste("xie_losses/figures_xie_losses/compare_misspec_NPMLEinit_", experiment_letter, ".png", sep="")
  ggsave(filename, plot)
}

for (experiment_letter in c("c", "d", "d5", "e", "f")){
  
  plot = df_mean %>%
    filter(data=="train", objective=="MSE", experiment == experiment_letter, 
           model != "NPMLEinit")  %>%
    mutate(location = ifelse(use_location, "Learn location", "Use median(Z)"),
           scale = ifelse(use_scale, "Learn scale", "Use IQR(Z)")) %>%
    ggplot(aes(x = factor(n))) +
    geom_point(aes(y = mean, color = model),) +
    geom_line(aes(y = mean, group = model, color=model)) +
    geom_errorbar(aes(ymin = mean-1.96*se, 
                      ymax = mean+1.96*se, 
                      group=model, color=model), alpha=0.5) +
    facet_wrap(scale ~ location) +
    ylab("In-sample MSE") +
    guides(alpha="none") +
    labs(title=paste("Experiment (", experiment_letter, ")", sep="")) +
    theme(axis.text.x = element_text(angle=45, hjust=1),
          legend.position="bottom") +
    xlab("log(n)")
  
  filename=paste("xie_losses/figures_xie_losses/", experiment_letter, "_MSE.png", sep="")
  ggsave(filename, plot)
  
}







#### Plot: use location, use scale ####

bayes_risk = data.frame("experiment" = c("c", "d", "d5", "e", "f"),
                        "yintercept" = c(0, 0, 0, 0.15, 0))

df_mean %>%
  filter(data=="train", objective=="SURE",
         use_location, use_scale) %>%
  filter(model != "NPMLEinit") %>%
  ggplot(aes(x = factor(n))) +
  geom_point(aes(y = mean, color = model),) +
  geom_line(aes(y = mean, group = model, color=model)) +
  geom_errorbar(aes(ymin = mean-1.96*se, 
                    ymax = mean+1.96*se, 
                    group=model, color=model), alpha=0.5) +
  geom_hline(data=bayes_risk, aes(yintercept=yintercept), alpha=0.6) +
  facet_wrap( ~ experiment, scales="free_y") +
  labs() +
  theme(axis.text.x = element_text(angle=45, hjust=1),
        legend.position="bottom",
        text = element_text(size=12)) +
  ylab("SURE") +
  xlab("log(n)")

ggsave("xie_losses/figures_xie_losses/location_scale_SURE.png", height=5, width=6)

df_mean %>%
  filter(data=="train", objective=="MSE",
         !use_location, use_scale) %>%
  filter(model != "NPMLEinit") %>%
  ggplot(aes(x = factor(n))) +
  geom_point(aes(y = mean, color = model),) +
  geom_line(aes(y = mean, group = model, color=model)) +
  geom_errorbar(aes(ymin = mean-1.96*se, 
                    ymax = mean+1.96*se, 
                    group=model, color=model), alpha=0.5) +
  geom_hline(data=bayes_risk, aes(yintercept=yintercept), alpha=0.6) +
  facet_wrap( ~ experiment, scales="free_y") +
  labs() +
  theme(axis.text.x = element_text(angle=45, hjust=1),
        legend.position="bottom",
        text = element_text(size=12)) +
  ylab("In-sample MSE") +
  xlab("log(n)")

ggsave("xie_losses/figures_xie_losses/location_scale_MSE.png", height=5, width=6)


#### Plot: use location, !use scale ####

df_mean %>%
  filter(data=="train", objective=="SURE",
         use_location, !use_scale) %>%
  # filter(model != "NPMLE") %>%
  ggplot(aes(x = factor(n))) +
  geom_point(aes(y = mean, color = model),) +
  geom_line(aes(y = mean, group = model, color=model)) +
  geom_errorbar(aes(ymin = mean-1.96*se, 
                    ymax = mean+1.96*se, 
                    group=model, color=model), alpha=0.5) +
  geom_hline(aes(yintercept=0)) +
  facet_wrap( ~ experiment, scales="free_y") +
  labs(title="Training objective, location only") +
  theme(axis.text.x = element_text(angle=45, hjust=1),
        legend.position="bottom") +
  ylab("SURE") +
  xlab("log(n)")

df_mean %>%
  filter(data=="train", objective=="MSE",
         use_location, !use_scale) %>%
  ggplot(aes(x = factor(n))) +
  geom_point(aes(y = mean, color = model),) +
  geom_line(aes(y = mean, group = model, color=model)) +
  geom_errorbar(aes(ymin = mean-1.96*se, 
                    ymax = mean+1.96*se, 
                    group=model, color=model), alpha=0.5) +
  geom_hline(aes(yintercept=0)) +
  facet_wrap( ~ experiment, scales="free_y") +
  labs(title="In-sample MSE, location only") +
  theme(axis.text.x = element_text(angle=45, hjust=1),
        legend.position="bottom") +
  ylab("In-sample MSE") +
  xlab("log(n)")

#### Plot: !use location, use scale ####

df_mean %>%
  filter(data=="train", objective=="SURE",
         !use_location, use_scale) %>%
  # filter(model != "NPMLE") %>%
  ggplot(aes(x = factor(n))) +
  geom_point(aes(y = mean, color = model),) +
  geom_line(aes(y = mean, group = model, color=model)) +
  geom_errorbar(aes(ymin = mean-1.96*se, 
                    ymax = mean+1.96*se, 
                    group=model, color=model), alpha=0.5) +
  geom_hline(aes(yintercept=0)) +
  facet_wrap( ~ experiment, scales="free_y") +
  labs(title="Training objective, scale only") +
  theme(axis.text.x = element_text(angle=45, hjust=1),
        legend.position="bottom") +
  ylab("SURE") +
  xlab("log(n)")

df_mean %>%
  filter(data=="train", objective=="MSE",
         !use_location, use_scale) %>%
  ggplot(aes(x = factor(n))) +
  geom_point(aes(y = mean, color = model),) +
  geom_line(aes(y = mean, group = model, color=model)) +
  geom_errorbar(aes(ymin = mean-1.96*se, 
                    ymax = mean+1.96*se, 
                    group=model, color=model), alpha=0.5) +
  geom_hline(aes(yintercept=0)) +
  facet_wrap( ~ experiment, scales="free_y") +
  labs(title="In-sample MSE, scale only") +
  theme(axis.text.x = element_text(angle=45, hjust=1),
        legend.position="bottom") +
  ylab("In-sample MSE") +
  xlab("log(n)")


## d, d5 with bfgs ##### 

# 25 simulations, 2 experiments, 3 values of n, train and test
xie_bfgs = read.csv("xie_dd5_bfgs.csv")
# xie_bfgs = xie_bfgs %>% bind_rows( read.csv("xie_dd5_bfgs_20000.csv"))
xie_bfgs_test = xie_bfgs %>% filter(data=="test")


sum(is.na(xie_bfgs_test))

xie_bfgs_test %>%
  mutate(is_na_wellspec = is.na(MSE_wellspec),
         is_na_misspec = is.na(MSE_misspec),
         has_na = is_na_wellspec | is_na_misspec) %>%
  group_by(n, experiment) %>% 
  summarize(total = n(),
            count_na = sum(has_na))

xie_bfgs_test %>%
  mutate(is_na_wellspec = is.na(MSE_wellspec),
         is_na_misspec = is.na(MSE_misspec)) %>%
  group_by(n, experiment, is_na_wellspec, is_na_misspec) %>% 
  summarize(total = n())

xie_bfgs_test %>%
  mutate(is_na_wellspec = is.na(MSE_wellspec),
         is_na_misspec = is.na(MSE_misspec),
         has_na = is_na_wellspec | is_na_misspec) %>%
  select(n, experiment, has_na, starts_with("SURE")) %>%
  pivot_longer(4:8, values_to = "SURE") %>%
  group_by(n, experiment, name, has_na) %>%
  summarize(mean_SURE = mean(SURE, na.rm=T), .groups="drop") %>%
  mutate(estimator = substr(name, 6, nchar(name))) %>%
  filter(estimator != "wellspec") %>%
  ggplot(aes(x = n, y = mean_SURE, color=estimator)) +
  geom_point() +
  geom_line() +
  facet_wrap(has_na~ experiment, scales="free_y") +
  labs(title="SURE, no covariates, by NA")

xie_bfgs_test %>%
  mutate(is_na_wellspec = is.na(MSE_wellspec),
         is_na_misspec = is.na(MSE_misspec),
         has_na = is_na_wellspec | is_na_misspec) %>%
  select(n, experiment, has_na, starts_with("SURE")) %>%
  pivot_longer(4:8, values_to = "SURE") %>%
  group_by(n, experiment, name) %>%
  summarize(mean_SURE = mean(SURE, na.rm=T), .groups="drop") %>%
  mutate(estimator = substr(name, 6, nchar(name))) %>%
  filter(estimator != "wellspec") %>%
  ggplot(aes(x = n, y = mean_SURE, color=estimator)) +
  geom_point() +
  geom_line() +
  facet_wrap(~ experiment, scales="free_y") +
  labs(title="SURE, no covariates, by NA")

xie_bfgs_test %>%
  mutate(is_na_wellspec = is.na(MSE_wellspec),
         is_na_misspec = is.na(MSE_misspec),
         has_na = is_na_wellspec | is_na_misspec) %>%
  select(n, experiment, has_na, starts_with("SURE")) %>%
  pivot_longer(4:8, values_to = "SURE") %>%
  group_by(n, experiment, name) %>%
  summarize(mean_SURE = mean(SURE, na.rm=T),
            se_SURE = sd(SURE, na.rm=T), .groups="drop")  %>% filter(experiment =="d")

ggsave("dd5_xie_sure_means.png")

xie_bfgs_test %>%
  mutate(is_na_wellspec = is.na(MSE_wellspec),
         is_na_misspec = is.na(MSE_misspec),
         has_na = is_na_wellspec | is_na_misspec) %>%
  select(n, experiment, has_na, starts_with("MSE")) %>%
  pivot_longer(4:8, values_to = "MSE") %>%
  group_by(n, experiment, name, has_na) %>%
  summarize(mean_MSE = mean(MSE, na.rm=T),
            se_MSE = sd(MSE, na.rm=T), .groups="drop") %>%
  mutate(estimator = substr(name, 5, nchar(name))) %>%
  filter(estimator != "wellspec") %>%
  ggplot(aes(x = n, y = mean_MSE, color=estimator)) +
  geom_point() +
  geom_line() +
  facet_wrap(has_na~ experiment, scales="free_y") +
  labs(title="Test MSE, no covariates, by NA")

ggsave("dd5_xie_mse_means.png")



## ADAM vs BFGS comparison ####

### Misspec ####
misspec = read.csv("adam_vs_bfgs_misspec.csv")

misspec_MSE = misspec %>% 
  select(experiment, starts_with("test")) %>%
  pivot_longer(ends_with("list")) %>%
  mutate(objective = ifelse(str_detect(name, "MSE"), "MSE", "SURE"),
         optimizer = ifelse(str_detect(name, "adam"), "Adam", "BFGS"),
         model = ifelse(str_detect(name, "misspec"), "EB misspecified", "EB NPMLEinit")) %>%
  filter(objective == "MSE") 

misspec %>% 
  select(experiment, starts_with("test_MSE")) %>%
  ggplot(aes(x = test_MSE_adam_misspec_list, y = test_MSE_adam_NPMLEinit_list)) +
  geom_point() + 
  geom_abline()  + 
  theme(aspect.ratio=1) +
  facet_wrap(~experiment, scales="free") + 
  labs(title="Adam")

misspec %>% 
  ggplot(aes(x = test_MSE_adam_NPMLEinit_list - test_MSE_adam_misspec_list )) +
  geom_histogram() + 
  geom_vline(xintercept=0)  + 
  theme(aspect.ratio=1) +
  facet_wrap(~experiment, scales="free")


misspec %>% 
  select(experiment, starts_with("test_MSE")) %>%
  ggplot(aes(x = test_MSE_bfgs_misspec_list, y = test_MSE_bfgs_NPMLEinit_list)) +
  geom_point() + 
  geom_abline() + 
  facet_wrap(~experiment, scales="free")+
  theme(aspect.ratio=1)  

### Wellspec ####

wellspec = read.csv("adam_vs_bfgs_wellspec.csv")

ggplot(a)



xie_bfgs_test %>%
  select(n, experiment, starts_with("SURE")) %>%
  pivot_longer(3:7, values_to = "SURE") %>%
  mutate(estimator = substr(name, 6, nchar(name))) %>%
  filter(SURE != -100) %>%
  filter(experiment=="d") %>%
  filter(estimator != "wellspec") %>%
  ggplot() +
  geom_boxplot(aes(x = estimator, y = SURE, group = estimator,
                   color = estimator)) +
  facet_wrap(experiment ~ n, nrow=1) +
  theme(axis.text.x = element_text(angle=45, vjust = 1, hjust=1),
        legend.position = "none", strip.text.x = element_text(size = 6),
        axis.text = element_text(size = 6)) +
  labs(title="Test SURE, no covariates")

ggsave("dd5_xie_sure_boxplot.png")

xie_bfgs_test %>%
  select(n, experiment, starts_with("MSE")) %>%
  pivot_longer(3:7, values_to = "MSE") %>%
  mutate(estimator = substr(name, 5, nchar(name))) %>%
  filter(MSE != -100) %>%
  filter(estimator != "wellspec", experiment=="d") %>%
  ggplot() +
  geom_boxplot(aes(x = estimator, y = MSE, group = estimator,
                   color = estimator)) +
  facet_wrap(experiment ~ n, nrow=1) +
  theme(axis.text.x = element_text(angle=45, vjust = 1, hjust=1),
        legend.position = "none", strip.text.x = element_text(size = 6),
        axis.text = element_text(size = 6)) + 
  labs(title="Test MSE, no covariates")

ggsave("dd5_xie_mse_boxplot.png")


ggplot(xie_bfgs_test, aes(x=MSE_wellspec)) + geom_histogram()
ggsave("dd5_xie_wellspec_mse.png")

ggplot(xie_bfgs_test, aes(x=SURE_wellspec)) + geom_histogram()
ggsave("dd5_xie_wellspec_sure.png")

## Compare NN sizes #####

compare_nn_sizes = read.csv("xie_losses/hidden_size_comparison.csv") %>%
  bind_rows(read.csv("xie_losses/hidden_size_comparison_single_layer.csv")) %>%
  mutate(different_run = (hidden_sizes %in% c("(8,)", "(32,)")))
path = "xie_losses/figures_xie_losses/"

group_minima = compare_nn_sizes %>%
  group_by(experiment, hidden_sizes, n, data) %>%
  summarize(med_MSE = median(MSE_wellspec),
            med_SURE = median(SURE_wellspec)) %>%
  group_by(experiment, n, data) %>%
  summarize(min_med_MSE = min(med_MSE),
            min_med_SURE = min(med_SURE))

compare_nn_sizes %>%
  filter(data=="train") %>%
  ggplot(aes(x = hidden_sizes, y = SURE_wellspec, group = hidden_sizes, fill = different_run)) +
  geom_boxplot(outliers=F) + 
  facet_wrap(n ~ experiment, nrow= 3, scales = "free") +
  theme(axis.text.x = element_text(angle=45, size=5),
        legend.position = "none") +
  ylab("Train SURE") +
  labs(subtitle="(8,), (32,) on different runs") +
  scale_fill_brewer(palette=2)

ggsave(paste(path, "comparison_train_SURE.png"))

compare_nn_sizes %>%
  filter(data=="test") %>%
  ggplot(aes(x = hidden_sizes, y = SURE_wellspec, group = hidden_sizes, fill = different_run)) +
  geom_boxplot(outliers=F) + 
  facet_wrap(n ~ experiment, nrow= 3, scales = "free") +
  theme(axis.text.x = element_text(angle=45, size=5),
        legend.position = "none") +
  ylab("Test SURE") +
  labs(subtitle="(8,), (32,) on different runs")+
  scale_fill_brewer(palette=2)

ggsave(paste(path, "comparison_test_SURE.png"))

compare_nn_sizes %>%
  filter(data=="train") %>%
  ggplot(aes(x = hidden_sizes, y = MSE_wellspec, group = hidden_sizes, fill = different_run)) +
  geom_boxplot(outliers=F) + 
  facet_wrap(n ~ experiment, nrow= 3, scales = "free") +
  theme(axis.text.x = element_text(angle=45, size=5),
        legend.position = "none") +
  ylab("Train MSE") +
  labs(subtitle="(8,), (32,) on different runs")+
  scale_fill_brewer(palette=2)

ggsave(paste(path, "comparison_train_MSE.png"))

compare_nn_sizes %>%
  filter(data=="test") %>%
  ggplot(aes(x = hidden_sizes, y = MSE_wellspec, group = hidden_sizes, fill = different_run)) +
  geom_boxplot(outliers=F) + 
  facet_wrap(n ~ experiment, nrow= 3, scales = "free") +
  theme(axis.text.x = element_text(angle=45, size=5),
        legend.position = "none") +
  ylab("Test MSE") +
  labs(subtitle="(8,), (32,) on different runs")+
  scale_fill_brewer(palette=2)

ggsave(paste(path, "comparison_test_MSE.png"))

## Small NN sizes for small n #####

path = "xie_losses/figures_xie_losses/"

df_names = list.files("xie_losses/hidden_layer_same_run_results/")
df_names = df_names[str_detect(df_names, "location_scale_small_hidden_sizes_skip_connect_comparison")]

compare_nn_sizes = NULL
for (df_name in df_names[-1]){ # ignore location_scale_comparison 0
  
  if (is.null(compare_nn_sizes)){
    compare_nn_sizes = read.csv(paste("xie_losses/hidden_layer_same_run_results", df_name, sep="/"))
  } else {
    compare_nn_sizes = compare_nn_sizes %>% bind_rows(read.csv(paste("xie_losses/hidden_layer_same_run_results", df_name, sep="/")))
  }
  
}

unique(compare_nn_sizes$hidden_sizes)
# ordered_nnsize_levels <- c("None", "(2,)", "(2, 2)", "(4,)", "(4, 4)", "(6,)", "(8,)", "(10,)")
compare_nn_sizes$hidden_sizes <- factor(compare_nn_sizes$hidden_sizes, levels = unique(compare_nn_sizes$hidden_sizes))

group_minima = compare_nn_sizes %>%
  group_by(experiment, hidden_sizes, n, data) %>%
  summarize(med_MSE = median(MSE_wellspec),
            med_SURE = median(SURE_wellspec)) %>%
  group_by(experiment, n, data) %>%
  summarize(min_med_MSE = min(med_MSE),
            min_med_SURE = min(med_SURE))


compare_nn_sizes %>%
  filter(data=="train") %>%
  ggplot(aes(x = hidden_sizes, y = SURE_wellspec, group = hidden_sizes, fill = experiment)) +
  geom_boxplot(outliers=F) + 
  facet_wrap(experiment ~ n, nrow= 2, scales = "free") +
  theme(axis.text.x = element_text(angle=45, size=5),
        legend.position = "none") +
  ylab("In-sample SURE") +
  labs(subtitle="Comparison with small n values and small NN sizes (skip connections)") +
  scale_fill_brewer(palette=2)

ggsave(paste(path, "comparison_smallnn_smalln_skipc_train_SURE.png"))

compare_nn_sizes %>%
  filter(data=="test") %>%
  ggplot(aes(x = hidden_sizes, y = SURE_wellspec, group = hidden_sizes, fill = experiment)) +
  geom_boxplot(outliers=F) + 
  facet_wrap(experiment ~ n, nrow= 2, scales = "free") +
  theme(axis.text.x = element_text(angle=45, size=5),
        legend.position = "none") +
  ylab("Out-of-sample SURE") +
  labs(subtitle="Comparison with small n values and small NN sizes (skip connections)")+
  scale_fill_brewer(palette=2)

ggsave(paste(path, "comparison_smallnn_smalln_skipc_test_SURE.png"))

compare_nn_sizes %>%
  filter(data=="train") %>%
  ggplot(aes(x = hidden_sizes, y = MSE_wellspec, group = hidden_sizes, fill = experiment)) +
  geom_boxplot(outliers=F) + 
  facet_wrap(experiment ~ n, nrow= 2, scales = "free") +
  theme(axis.text.x = element_text(angle=45, size=5),
        legend.position = "none") +
  ylab("In-sample MSE") +
  labs(subtitle="Comparison with small n values and small NN sizes (skip connections)")+
  scale_fill_brewer(palette=2)

ggsave(paste(path, "comparison_smallnn_smalln_skipc_train_MSE.png"))

compare_nn_sizes %>%
  filter(data=="test") %>%
  ggplot(aes(x = hidden_sizes, y = MSE_wellspec, group = hidden_sizes, fill = experiment)) +
  geom_boxplot(outliers=F) + 
  facet_wrap(experiment ~ n, nrow= 2, scales = "free") +
  theme(axis.text.x = element_text(angle=45, size=5),
        legend.position = "none") +
  ylab("Out-of-sample MSE") +
  labs(subtitle="Comparison with small n values and small NN sizes (skip connections)")+
  scale_fill_brewer(palette=2)

ggsave(paste(path, "comparison_smallnn_smalln_skipc_test_MSE.png"))

## Different NN sizes for high n #####

path = "xie_losses/figures_xie_losses/"

df_names = list.files("xie_losses/hidden_layer_same_run_results/")
df_names = df_names[str_detect(df_names, "location_scale_small_hidden_sizes_highn_comparison")]

compare_nn_sizes = NULL
for (df_name in df_names[-1]){ # ignore location_scale_comparison 0
  
  if (is.null(compare_nn_sizes)){
    compare_nn_sizes = read.csv(paste("xie_losses/hidden_layer_same_run_results", df_name, sep="/"))
  } else {
    compare_nn_sizes = compare_nn_sizes %>% bind_rows(read.csv(paste("xie_losses/hidden_layer_same_run_results", df_name, sep="/")))
  }
  
}

unique(compare_nn_sizes$hidden_sizes)
ordered_nnsize_levels <- c('None', '(2,)', '(2,2)', '(4,)', '(4,4)', '(8,)', '(16,)', '(16,16)', '(32,)', '(64,)')
compare_nn_sizes$hidden_sizes <- factor(compare_nn_sizes$hidden_sizes, levels = unique(compare_nn_sizes$hidden_sizes))

group_minima = compare_nn_sizes %>%
  group_by(experiment, hidden_sizes, n, data) %>%
  summarize(med_MSE = median(MSE_wellspec),
            med_SURE = median(SURE_wellspec)) %>%
  group_by(experiment, n, data) %>%
  summarize(min_med_MSE = min(med_MSE),
            min_med_SURE = min(med_SURE))


compare_nn_sizes %>%
  filter(data=="train") %>%
  ggplot(aes(x = hidden_sizes, y = SURE_wellspec, group = hidden_sizes, fill = experiment)) +
  geom_boxplot(outliers=F) + 
  facet_wrap(experiment ~ n, nrow= 2, scales = "free") +
  theme(axis.text.x = element_text(angle=45, size=5),
        legend.position = "none") +
  ylab("Train SURE") +
  labs(subtitle="Comparison with high n values and different NN sizes") +
  scale_fill_brewer(palette=2)

ggsave(paste(path, "comparison_diffnn_highn_train_SURE.png"))

compare_nn_sizes %>%
  filter(data=="test") %>%
  ggplot(aes(x = hidden_sizes, y = SURE_wellspec, group = hidden_sizes, fill = experiment)) +
  geom_boxplot(outliers=F) + 
  facet_wrap(experiment ~ n, nrow= 2, scales = "free") +
  theme(axis.text.x = element_text(angle=45, size=5),
        legend.position = "none") +
  ylab("Test SURE") +
  labs(subtitle="Comparison with high n values and different NN sizes")+
  scale_fill_brewer(palette=2)

ggsave(paste(path, "comparison_diffnn_highn_test_SURE.png"))

compare_nn_sizes %>%
  filter(data=="train") %>%
  ggplot(aes(x = hidden_sizes, y = MSE_wellspec, group = hidden_sizes, fill = experiment)) +
  geom_boxplot(outliers=F) + 
  facet_wrap(experiment ~ n, nrow= 2, scales = "free") +
  theme(axis.text.x = element_text(angle=45, size=5),
        legend.position = "none") +
  ylab("Train MSE") +
  labs(subtitle="Comparison with high n values and different NN sizes")+
  scale_fill_brewer(palette=2)

ggsave(paste(path, "comparison_diffnn_highn_train_MSE.png"))

compare_nn_sizes %>%
  filter(data=="test") %>%
  ggplot(aes(x = hidden_sizes, y = MSE_wellspec, group = hidden_sizes, fill = experiment)) +
  geom_boxplot(outliers=F) + 
  facet_wrap(experiment ~ n, nrow= 2, scales = "free") +
  theme(axis.text.x = element_text(angle=45, size=5),
        legend.position = "none") +
  ylab("Test MSE") +
  labs(subtitle="Comparison with high n values and different NN sizes")+
  scale_fill_brewer(palette=2)

ggsave(paste(path, "comparison_diffnn_highn_test_MSE.png"))

## MSE for experiment i #####

library(latex2exp)
library(RColorBrewer)
library(lemon)
library(scales)
palette1 <- brewer.pal(8, "Dark2")
palette2 <- brewer.pal(8, "Set1")
palette3 <- brewer.pal(8, "Set2")
dark2_palette <- unique(c(palette1, palette2, palette3)) 
theme_set(theme_bw())

df_names = list.files("xie_losses/hidden_layer_same_run_results/")
df_names = df_names[str_detect(df_names, "location_scale_i_small_hidden_sizes_comparison")]

df = NULL
for (df_name in df_names[-1]){ # ignore location_scale_comparison 0
  
  if (is.null(df)){
    df = read.csv(paste("xie_losses/hidden_layer_same_run_results", df_name, sep="/"))
  } else {
    df = df %>% bind_rows(read.csv(paste("xie_losses/hidden_layer_same_run_results", df_name, sep="/")))
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

#### Plot: !use location, use scale ####

m_sim = unique(df_mean$count)

bayes_risk = data.frame("experiment" = c("i"),
                        "yintercept" = c(1.327),
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
         experiment = factor(experiment, levels=c("i"))) 
# experiment = factor(experiment, levels=c("c", "d", "e", "f"))) 

ground_truth_df = df_mean_plot %>% 
  group_by(n, experiment) %>%
  summarize(count = n()) %>% 
  left_join(bayes_risk, by = "experiment") %>%
  mutate(model = factor(model, levels=c("NPMLE", "SURE-PM", "SURE-grandmean", "SURE-THING", "EBCF", "SURE-LS", "Bayes risk")),
         experiment = factor(experiment)) %>%
  rename(mean = yintercept)


levels(df_mean_plot$experiment) = c(i = TeX(r"(Five$$ covariates)"))

levels(ground_truth_df$experiment) = c(i = TeX(r"(Five$$ covariates)"))

levels(bayes_risk$experiment) = c(i = TeX(r"(Five$$ covariates)"))

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
  facet_wrap( ~ experiment, scales="free_y", labeller=label_parsed, nrow = 1, ncol = 1) +
  # labs(title="In-sample MSE, scale only") +
  theme(axis.text.x = element_text(angle=45, hjust=1),
        legend.position="bottom",
        text=element_text(size=15),
        legend.key.size = unit(2,"line")) +
  # geom_hline(data = bayes_risk, aes(yintercept = yintercept), show.legend = F) +
  geom_text(data = bayes_risk, aes(x = 5, y = yintercept, label = paste0("Bayes risk: ", yintercept)), hjust = -0.15, vjust = 1.35, color = 'black', size = 7) +
  scale_y_continuous(expand = expansion(mult = c(0.13, 0.1))) + 
  scale_color_manual(values=dark2_palette[c(1,3,2,4,10,9,8)], name="") +
  scale_linetype_manual(values=c(2, 4, 3, 1, 6, 5, 1), name="") +
  scale_shape_manual(values=c(15, 17, 16, 18, 19, 20, 1), name="") +
  labs(subtitle="Hidden size = (2,2)") +
  ylab("In-sample MSE") +
  xlab("n")

## MSE for experiment e #####

df_names = list.files("xie_losses/hidden_layer_same_run_results/")
df_names = df_names[str_detect(df_names, "location_scale_c_no_hidden_sizes_comparison")]

df = NULL
for (df_name in df_names[-1]){ # ignore location_scale_comparison 0
  
  if (is.null(df)){
    df = read.csv(paste("xie_losses/hidden_layer_same_run_results", df_name, sep="/"))
  } else {
    df = df %>% bind_rows(read.csv(paste("xie_losses/hidden_layer_same_run_results", df_name, sep="/")))
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

#### Plot: !use location, use scale ####

m_sim = unique(df_mean$count)

bayes_risk = data.frame("experiment" = c("c"),
                        "yintercept" = c(0),
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
         experiment = factor(experiment, levels=c("c"))) 
# experiment = factor(experiment, levels=c("c", "d", "e", "f"))) 

ground_truth_df = df_mean_plot %>% 
  group_by(n, experiment) %>%
  summarize(count = n()) %>% 
  left_join(bayes_risk, by = "experiment") %>%
  mutate(model = factor(model, levels=c("NPMLE", "SURE-PM", "SURE-grandmean", "SURE-THING", "EBCF", "SURE-LS", "Bayes risk")),
         experiment = factor(experiment)) %>%
  rename(mean = yintercept)


levels(df_mean_plot$experiment) = c(c = TeX(r"(Uniform$$ prior)"))

levels(ground_truth_df$experiment) = c(c = TeX(r"(Uniform$$ prior)"))

levels(bayes_risk$experiment) = c(c = TeX(r"(Uniform$$ prior)"))

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
  facet_wrap( ~ experiment, scales="free_y", labeller=label_parsed, nrow = 1, ncol = 1) +
  # labs(title="In-sample MSE, scale only") +
  theme(axis.text.x = element_text(angle=45, hjust=1),
        legend.position="bottom",
        text=element_text(size=15),
        legend.key.size = unit(2,"line")) +
  # geom_hline(data = bayes_risk, aes(yintercept = yintercept), show.legend = F) +
  geom_text(data = bayes_risk, aes(x = 5, y = yintercept, label = paste0("Bayes risk: ", yintercept)), hjust = -0.15, vjust = 1.35, color = 'black', size = 7) +
  scale_y_continuous(expand = expansion(mult = c(0.13, 0.1))) + 
  scale_color_manual(values=dark2_palette[c(1,3,2,4,10,9,8)], name="") +
  scale_linetype_manual(values=c(2, 4, 3, 1, 6, 5, 1), name="") +
  scale_shape_manual(values=c(15, 17, 16, 18, 19, 20, 1), name="") +
  labs(subtitle="Hidden size = None") +
  ylab("In-sample MSE") +
  xlab("n")


## Checking for unbiasedness of SURE for MSE ####

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

df_wellspec = df %>%
  filter(model == "wellspec",
         !use_location, use_scale) %>%
  mutate(experiment = factor(experiment),
         experiment = factor(experiment, levels=c("c", "d", "e", "f", "g", "h", "i", "j"))) %>%
  select(n, data, experiment, objective, model, value) 

df_wellspec %>%
  filter(experiment=='c') %>%
  ggplot(aes(x = factor(n), y = value, group = n, fill = objective)) +
  geom_boxplot(outliers=F) + 
  facet_wrap(data ~ objective, nrow= 2) +
  theme(axis.text.x = element_text(angle=45, size=5),
        legend.position = "none") +
  ylab("SURE/MSE") +
  xlab("n") + 
  labs(subtitle=paste("Experiment c - wellspecified")) +
  scale_fill_brewer(palette=2)

ggsave(paste(path, "comparison_MSE_SURE_c.png")) 

