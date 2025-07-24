library(tidyverse)
library(ggh4x) # facetted_pos_scales
library(ggpubr) # arrange multiple figures together

## No covariates: MSE and SURE ####

### Normal theta case (Homoscedastic) ####

nocovariates_normal  = NULL

for (file in list.files("results/homoscedastic", pattern="normal_homo_*")){
  
  file_name = paste0("results/homoscedastic/", file)
  
  if (is.null(nocovariates_normal)){
    
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

### Binary theta case (Homoscedastic) ####

nocovariates_binary  = NULL

for (file in list.files("results/homoscedastic", pattern="binary_homo_*")){
  
  file_name = paste0("results/homoscedastic/", file)
  
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
  filter(estimator_name %in% c("SURE_both", "NPMLE", "BAYES") ) %>%
  group_by(mu, k, estimator_name) %>%
  summarize(mean_sum_squared_errors = mean(MSE),
            count = n()) %>% view() 
