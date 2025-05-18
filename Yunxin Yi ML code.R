# Setup ------------------------------------------------------------------------
# Load libraries
library(tidyverse)
library(arrow)    # for reading parquet
library(haven)    # for reading .dta (Stata)
library(lubridate) # for date handling
library(future)
library(rsample)
library(tidymodels)
library(dbplyr)
library(RPostgres)
library(DBI)
library(glue)
library(haven)
library(tidyverse) 
# Log into wrds ----------------------------------------------------------------
keyring::key_set("WRDS_USER")
keyring::key_set("WRDS_PW")

if(exists("wrds")){
  dbDisconnect(wrds)  
}
wrds <- dbConnect(Postgres(),
                  host = 'wrds-pgdata.wharton.upenn.edu',
                  port = 9737,
                  user = keyring::key_get("WRDS_USER"),
                  password = keyring::key_get("WRDS_PW"),
                  sslmode = 'require',
                  dbname = 'wrds')
library(DBI)

# Query from wrdsapps_finratio.firm_ratio
query <- "
SELECT gvkey, adate, qdate, dpr, efftax
FROM wrdsapps_finratio.firm_ratio
WHERE adate BETWEEN '1978-01-01' AND '2018-12-31'
"


# Run query
tax_div_data <- dbGetQuery(wrds, query)

# Setup data path
usethis::edit_r_environ()
data_path <- Sys.getenv("data_path")  # or use getOption("data_path")


# Load main dataset
main_data <- read_parquet(glue("{data_path}/data_M2.parquet"))


# Merging data
tax_div2 <- tax_div_data |> 
  filter(adate==qdate) |> 
  select(gvkey,adate,dpr,efftax) |> 
  distinct()

tax_div2 |> 
  group_by(gvkey,adate) |> 
  count() |> 
  arrange(-n)


merged_data <- main_data %>%
  left_join(tax_div2, by = c("gvkey", "datadate" = "adate"))


# Select only the needed variables and apply filters
final_data <- merged_data %>%
  filter(
    year(datadate) >= 1978,
    abs(EARN) < 1,
    abs(EARN_lag_1) < 1,
    abs(EARN_lead_1) < 1,
    MVE > 10,
    dpr >= 0,
    dpr<1 ,
    abs(efftax) < 1
  ) %>%
  select(
    gvkey, datadate, 
    EARN_lead_1, EARN, EARN_lag_1,
    ACC_HVZ, PM, ATO, LEV, SG, 
    ACC_HVZ, AT_HVZ, D_HVZ, LOSS, MVE,
    PM_lag_1, ATO_lag_1, LEV_lag_1, SG_lag_1, 
    ACC_HVZ_lag_1, AT_HVZ_lag_1, D_HVZ_lag_1, LOSS_lag_1, MVE_lag_1,
    dpr, efftax
  )|> 
  drop_na()

# Table 2 descriptive statistics
summary(final_data)


# Parallel processing setup ----------------------------------------------------
configure_parallel_xgb <- function(total_cores = parallel::detectCores(), 
                                   reserve_cores = 2, 
                                   desired_workers = NULL) {
  available_cores <- total_cores - reserve_cores
  if (is.null(desired_workers)) {
    desired_workers <- min(available_cores %/% 16, 8)
  }
  threads_per_worker <- floor(available_cores / desired_workers)
  future::plan(multisession, workers = desired_workers)
  message(glue::glue(
    "✅ Parallel plan set with {desired_workers} workers and {threads_per_worker} threads per XGBoost model.\n",
    "🧠 Total used cores: {desired_workers * threads_per_worker} out of {total_cores}."
  ))
  return(threads_per_worker)
}
xgb_threads <- configure_parallel_xgb(total_cores = 96)
options(xgb_threads = xgb_threads)


# Subset for tuning
tuning_data <- final_data |> 
  filter(year(datadate) >= 1978,
         year(datadate) <= 1994)




#Select a subset of the data to use for tuning ---------------------------------


# Subset for tuning
tuning_data <- final_data |> 
  filter(year(datadate) >= 1978,
         year(datadate) <= 1994)




# Step 3: Define resamples (3-fold CV for tuning)
# ChatGPT recommends 3 folds for Bayesian tuning with enough data
cv_folds <- vfold_cv(tuning_data, v = 3)


#Define two pre-processing recipes ---------------------------------------------

#Let's rename our original recipe the "noint" recipe for no interactions
noint_recipe <- recipe(x = tuning_data |> dplyr::slice(1)) |> 
  update_role(gvkey,datadate, new_role = "ID") |> 
  update_role(EARN_lead_1, new_role = "outcome") |>
  update_role(EARN , EARN_lag_1 , ACC_HVZ , PM , ATO , 
              LEV , SG , ACC_HVZ , AT_HVZ , D_HVZ , LOSS , MVE ,
              PM_lag_1 , ATO_lag_1 , LEV_lag_1 , SG_lag_1 , 
              ACC_HVZ_lag_1 , AT_HVZ_lag_1 , D_HVZ_lag_1 , 
              LOSS_lag_1 , MVE_lag_1 , dpr, efftax, 
              new_role = "predictor") |> 
  step_rm(has_role(NA)) |> 
  step_naomit(everything(), skip = TRUE) |> 
  step_normalize(all_predictors()) 

#let's make a second recipe to evaluate whether there are useful interactions
int_recipe <- recipe(x = tuning_data |> dplyr::slice(1)) |> 
  update_role(gvkey,datadate, new_role = "ID") |> 
  update_role(EARN_lead_1, new_role = "outcome") |>
  update_role(EARN , EARN_lag_1 , ACC_HVZ , PM , ATO , 
              LEV , SG , ACC_HVZ , AT_HVZ , D_HVZ , LOSS , MVE ,
              PM_lag_1 , ATO_lag_1 , LEV_lag_1 , SG_lag_1 , 
              ACC_HVZ_lag_1 , AT_HVZ_lag_1 , D_HVZ_lag_1 , 
              LOSS_lag_1 , MVE_lag_1 , dpr, efftax, 
              new_role = "predictor") |>  
  step_rm(has_role(NA)) |> 
  step_naomit(everything(), skip = TRUE) |> 
  step_normalize(all_predictors()) |> 
  step_interact(~ all_predictors():all_predictors())

#Define four potential models --------------------------------------------------

#Model 1 - Simple OLS
ols_spec <- 
  linear_reg()  |>  
  set_engine("lm")

#Model 2 - Untuned XGBoost
# XGBoost with Huber loss
xgb0_spec <- boost_tree() |>
  set_engine("xgboost",
             objective = "reg:pseudohubererror",
             eval_metric = "mae",
             early_stopping_rounds = 10,
             nthread = getOption("xgb_threads")) |>
  set_mode("regression")

#Model 3 - Elastic Net (Basically a tuned regression with regularization)
enet_spec <- linear_reg(
  penalty = tune(),
  mixture = tune()
) |>
  set_engine("glmnet") |>
  set_mode("regression")

#Model 4 - Tunable XGBoost
xgb_spec <- boost_tree(
  trees = tune(),
  learn_rate = tune(),
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  mtry = tune()
) |>
  set_engine("xgboost",
             objective = "reg:pseudohubererror",
             eval_metric = "mae",
             early_stopping_rounds = 10,
             nthread = getOption("xgb_threads")) |>
  set_mode("regression")

# Model 5 - rifge regresser
ridge_spec <- linear_reg(
  penalty = tune(),     # λ will be tuned
  mixture = 0           # 0 = Ridge, 1 = Lasso, in-between = Elastic Net
) |>
  set_engine("glmnet") |>
  set_mode("regression")



# make a workflow set ----------------------------------------------------------

# 2 recipes x 5 models = a set of 10 workflows
wf_set <- workflow_set(
  preproc = list("noint" = noint_recipe,
                 "int" = int_recipe),
  models = list('xgb_tuned' = xgb_spec,
                'xgb_notune' = xgb0_spec,
                'OLS' = ols_spec,
                'enet' = enet_spec,
                'ridge' = ridge_spec)
)




# Define parameters for tuning -------------------------------------------------

#parameter ranges to explore for elastic net
enet_params <- parameters(
  penalty(range = c(-4, 0)),  # log10 scale: 1e-4 to 1
  mixture(range = c(0, 1))
)

#parameter ranges to explore for XGBoost
xgb_params <- parameters(
  finalize(mtry(), tuning_data),
  trees(range = c(100L, 1000L)),
  learn_rate(range = c(-3, -0.5)),       # log10 scale: 0.001 to ~0.3
  tree_depth(range = c(3L, 10L)),
  min_n(range = c(2L, 20L)),
  loss_reduction(range = c(-3, 1))       # log10 scale: 0.001 to 10
)

#parameter ranges to explore for ridge
ridge_params <- parameters(
  penalty(range = c(-4, 0))  # log10 scale: 1e-4 to 1
)


# Metric for tuning
mae_metric <- metric_set(mae)

#Add the parameter sets to each relevant workflow
wf_set <- wf_set |>
  option_add(param_info = enet_params, id = "noint_enet") |>
  option_add(param_info = enet_params, id = "int_enet") |>
  option_add(param_info = xgb_params, id = "noint_xgb_tuned") |>
  option_add(param_info = xgb_params, id = "int_xgb_tuned") |>
  option_add(param_info = ridge_params, id = "noint_ridge") |>
  option_add(param_info = ridge_params, id = "int_ridge")

# trained <- prep(noint_recipe, training(cv_folds$splits[[1]]))
# train_baked <- bake(trained, new_data = NULL)
# test_baked <- bake(trained, new_data = testing(cv_folds$splits[[1]]))
# 
# dim(train_baked)
# dim(test_baked)


# Run Bayesian tuning using workflow_map ---------------------------------------

# Note that the plan() for future was set above in the configure function
#It might be good to try changing kappa to 2 or to try it 
#without the objective = line
#if you set verbose = TRUE you can see it working
#but it will produce a lot of output 
tuned_results <- wf_set |>
  workflow_map(
    fn = "tune_bayes",
    resamples = cv_folds,
    metrics = mae_metric,
    initial = 20,
    iter = 50,
    seed = 1234,
    verbose = TRUE,
    objective = conf_bound(kappa = 3.0),
    control = control_bayes(
      verbose = FALSE,
      no_improve = 20,
      save_workflow = TRUE,
      save_pred = TRUE,
      parallel_over = "everything"
    )
  )

# Save tuning results
saveRDS(tuned_results, "C:/Users/y371y552/Desktop/tuned_results_3.rds")
tuned_results <- readRDS("C:/Users/y371y552/Desktop/tuned_results_3.rds")

# Visualize our Tuning results
# Figure 1 panel A
tuned_results |>
  collect_metrics(summarize = TRUE) |>
  filter(.metric == "mae") |>
  group_by(wflow_id) |>
  slice_min(order_by = mean, n = 1) |>
  ungroup() |>
  select(wflow_id, mean) |> 
  print()

fixed_metrics <- tuned_results |>
  collect_metrics(summarize = TRUE) |>
  filter(.metric == "mae") |>
  group_by(wflow_id) |>
  slice_min(order_by = mean, n = 1) |>
  ungroup() |>
  mutate(mean = as.numeric(mean))  # Just in case

ggplot(fixed_metrics, aes(x = reorder(wflow_id, mean), y = mean)) +
  geom_col(fill = "steelblue", width = 0.7) +
  geom_text(aes(label = sprintf("%.3f", mean)),
            hjust = 1.1, color = "white", size = 4) +
  coord_flip() +
  labs(title = "Best MAE per Workflow", x = "Workflow", y = "MAE") +
  theme_minimal()

# Load libraries
library(dplyr)
library(ggplot2)

# Step 1: Clean and extract best MAE per workflow
fixed_metrics <- tuned_results |>
  collect_metrics(summarize = TRUE) |>
  filter(.metric == "mae") |>
  group_by(wflow_id) |>
  slice_min(order_by = mean, n = 1) |>  # Keep only the best per workflow
  ungroup() |>
  distinct(wflow_id, .keep_all = TRUE) |>  # Remove any remaining duplicates
  mutate(mean = as.numeric(mean),          # Ensure numeric
         label_str = sprintf("%.4f", mean))  # Format label

# Step 2: Plot
ggplot(fixed_metrics, aes(x = reorder(wflow_id, mean), y = mean)) +
  geom_col(fill = "steelblue", width = 0.7) +
  geom_text(aes(label = label_str), 
            hjust = 1.1, color = "white", size = 4) +
  coord_flip() +
  labs(title = "Best MAE per Workflow", x = "Workflow", y = "MAE") +
  theme_minimal()






#version from the tidy modelling with R book
autoplot(
  tuned_results,
  rank_metric = "mae",  # <- how to order models
  metric = "mae",       # <- which metric to visualize
  select_best = TRUE     # <- one point per workflow
) +
  geom_text(aes(y = mean - .01, label = wflow_id), angle = 90, hjust = 1) +
  lims(y = c(.02, .09)) +
  theme_bw() + 
  theme(legend.position = "none") 

#this collects the tuning data from the xgboost
xgb_tune_data <- tuned_results |> 
  filter(wflow_id == "int_xgb_tuned") |> 
  pull(result) |>
  (\(x) x[[1]])() |>
  collect_metrics(summarize = TRUE)

#plot how accuracry evolves over the tuning process
xgb_tune_data |>
  filter(.iter > 0) |> 
  ggplot(aes(x = .iter, y = mean)) +
  geom_line() +
  geom_point() +
  labs(title = "MAE over Bayesian iterations (noint_xgb_tuned)",
       x = "Iteration",
       y = "MAE")


#same plot but for elastic net
enet_tune_data <- tuned_results |> 
  filter(wflow_id == "int_enet") |> 
  pull(result) |>
  (\(x) x[[1]])() |>
  collect_metrics(summarize = TRUE)

enet_tune_data |> 
  filter(.iter > 0) |> 
  ggplot(aes(x = .iter, y = mean)) +
  geom_line() +
  geom_point() +
  labs(title = "MAE over Bayesian iterations (int_enet)",
       x = "Iteration",
       y = "MAE")

# Extract tuning results for ridge regression
ridge_tune_data <- tuned_results |> 
  filter(wflow_id == "int_ridge") |> 
  pull(result) |>
  pluck(1) |> 
  collect_metrics(summarize = TRUE)

# Plot MAE over iterations for Ridge
ridge_tune_data |> 
  filter(.iter > 0) |> 
  ggplot(aes(x = .iter, y = mean)) +
  geom_line() +
  geom_point() +
  labs(
    title = "MAE over Bayesian iterations (int_ridge)",
    x = "Iteration",
    y = "MAE"
  ) +
  theme_minimal()




# Finalize workflows using best params (or leave as-is if not tuned)

best_params  <- map(tuned_results$wflow_id, ~ tuned_results |>
                      extract_workflow_set_result(.x) |>   # get tuning result
                      select_best(metric = "mae")          # get best parameters
)

best_params


final_wf = map2(tuned_results$wflow_id, best_params, ~
                  tuned_results |>
                  extract_workflow(.x) |>             # get original workflow
                  finalize_workflow(.y)               # plug in best params
)

final_wf


wf_list_named <- set_names(final_wf, tuned_results$wflow_id)

# Convert to a workflow_set
final_wf_set <- as_workflow_set(!!!wf_list_named)



# now lets apply the finalized models to an expanding window -------------------

# create a resampling object for the expanding window
# this object slides forward one year at a time
# it uses all prior data as analysis data and one year ahead as assessment
expand_window <- sliding_period(
  data = final_data |> 
    filter(year(datadate) >= 1978, year(datadate) <= 2018) |> 
    arrange(datadate),
  index = datadate,
  period = "year",
  skip = 0,
  lookback = Inf
)

#visualize the expanding window ------------------------------------------------

# Extract info from each split
expand_vis <- tibble(
  id = expand_window$id,
  split = expand_window$splits
) |>
  mutate(
    row = as.integer(str_remove(id, "Slice")),
    train = map(split, ~ tibble(
      year = year(training(.x)$datadate),
      type = "train"
    )),
    test = map(split, ~ tibble(
      year = year(testing(.x)$datadate),
      type = "test"
    ))
  ) |>
  select(row, train, test) |>
  pivot_longer(cols = c(train, test), values_to = "data") |>
  unnest(data)

# plot expanding window
ggplot(expand_vis, aes(x = year, y = row, fill = type)) +
  geom_tile() +
  scale_fill_manual(values = c(train = "skyblue", test = "tomato")) +
  scale_x_continuous(breaks = seq(1978, 2018, by = 4)) +
  labs(
    title = "Expanding Window Cross-Validation",
    x = "Year",
    y = "Resample Index",
    fill = NULL
  ) +
  theme_minimal()

# Apply the tuned models to the expanding window -------------------------------

window_results <- final_wf_set |>
  workflow_map(
    fn = "fit_resamples",
    resamples = expand_window,
    metrics = metric_set(mae),
    control = control_resamples(
      save_pred = TRUE,
      verbose = TRUE,
      parallel_over = "everything"  # parallelize folds *and* models
    )
  )
# Save and load window results 
saveRDS(window_results, "C:/Users/y371y552/Desktop/window_results_1.rds")
window_results <- readRDS("C:/Users/y371y552/Desktop/window_results_1.rds")

# Evaluate Performance ---------------------------------------------------------
#Visualize results

#need row ids to link in year
row_ids <- final_data |>
  filter(year(datadate) >= 1998, year(datadate) <= 2018) |> 
  arrange(datadate) |> 
  mutate(.row = row_number(),
         year = year(datadate)) |> 
  select(.row,year)

# Now join it to your predictions
preds_with_id <- window_results |>
  collect_predictions() |>
  left_join(row_ids, by = ".row") |> 
  filter(!is.na(.pred)) |> 
  #only keep observations with all forecasts
  group_by(.row) |> 
  filter(n()==10) |> 
  ungroup()
# Figure 1 Panel B
# Plot Out-of-Sample MAE by Year
performance_over_time <- preds_with_id |>
  group_by(wflow_id, year) |>
  summarise(mae = mean(abs(.pred - EARN_lead_1), na.rm=T), .groups = "drop")

performance_over_time <- performance_over_time |>
  filter(year < 2018)

ggplot(performance_over_time, aes(x = year, y = mae, color = wflow_id, group = wflow_id)) +
  geom_line() +
  geom_point() +
  labs(
    title = "Out-of-Sample MAE by Year",
    x = "Year",
    y = "Mean Absolute Error",
    color = "Model"
  ) +
  theme_minimal()
# Plot Cumulative MAE over time
performance_over_time |>
  group_by(wflow_id) |>
  arrange(year) |>
  mutate(cum_mae = cumsum(mae)) |>
  ggplot(aes(x = year, y = cum_mae, color = wflow_id)) +
  geom_line() +
  labs(title = "Cumulative MAE (Expanding Window)", y = "Cumulative MAE") +
  theme_minimal()

#leaderboard
performance_over_time |>
  group_by(wflow_id) |>
  arrange(year) |>
  summarise(cum_mae = sum(mae), .groups = "drop") |>
  arrange(cum_mae)

#median absolute error instead of mean
 preds_with_id |>
  group_by(wflow_id, year) |>
  summarise(mae = median(abs(.pred - EARN_lead_1), na.rm=T), 
            .groups = "drop") |> 
   group_by(wflow_id) |>
   arrange(year) |>
   summarise(cum_mae = sum(mae), .groups = "drop") |>
   arrange(cum_mae)

 expand_summary <- tibble(
   id = expand_window$id,
   split = expand_window$splits
 ) |>
   mutate(
     test_data = map(split, ~ testing(.x)),
     year = map_int(test_data, ~ year(.x$datadate[1])),
     n_obs = map_int(test_data, nrow)
   ) |>
   select(year, n_obs)

 expand_summary
 
 #Let conduct some local model explication
 library(DALEXtra)
 library(tidymodels)
 
 vip_feature <- c(
   "EARN", "EARN_lag_1", "ACC_HVZ", "PM", "ATO", "LEV", "SG", "AT_HVZ", "D_HVZ", "LOSS", "MVE",
   "PM_lag_1", "ATO_lag_1", "LEV_lag_1", "SG_lag_1", "ACC_HVZ_lag_1", "AT_HVZ_lag_1", 
   "D_HVZ_lag_1", "LOSS_lag_1", "MVE_lag_1",  "dpr", "efftax"
 )
 
 vip_train <- tuning_data |> select(all_of(vip_feature))
 
 # Extract best parameters for "noint_ridge" 
 best_ridge_params <- tuned_results |>
   extract_workflow_set_result("noint_ridge") |>
   select_best(metric = "mae")
 
 # Finalize the ridge_spec with best parameters
 ridge_final_spec <- finalize_model(ridge_spec, best_ridge_params)
 
 
 # Fit XGBoost, OLS and ridge
 xgb_fit <- workflow() |>
   add_model(xgb0_spec) |>
   add_formula(EARN_lead_1 ~ .) |>
   fit(data = bind_cols(vip_train, EARN_lead_1 = tuning_data$EARN_lead_1))
 
 ols_fit <- workflow() |>
   add_model(ols_spec) |>
   add_formula(EARN_lead_1 ~ .) |>
   fit(data = bind_cols(vip_train, EARN_lead_1 = tuning_data$EARN_lead_1))
 
 ridge_fit <- workflow() |>
   add_model(ridge_final_spec) |>
   add_formula(EARN_lead_1 ~ .) |>
   fit(data = bind_cols(vip_train, EARN_lead_1 = tuning_data$EARN_lead_1) |> drop_na())
 
 
 
 # Explain
 explainer_xgb <- explain(
   xgb_fit,
   data = vip_train,
   y = tuning_data$EARN_lead_1,
   label = "XGBoost"
 )
 
 explainer_ols <- explain(
   ols_fit,
   data = vip_train,
   y = tuning_data$EARN_lead_1,
   label = "OLS"
 )
 
 explainer_ridge <- explain(
   ridge_fit,
   data = vip_train,
   y = tuning_data$EARN_lead_1,
   label = "Ridge",
   type = "regression"
 )
 
# Local explanation (e.g. for first observation)
xgb_breakdown <- predict_parts(explainer_xgb, new_observation = vip_train[1, ], type = "shap")
plot(xgb_breakdown)
 
lm_breakdown <- predict_parts(explainer_ols, new_observation = vip_train[1, ], type = "break_down")
lm_breakdown

ridge_breakdown <- predict_parts(explainer_ridge, new_observation = vip_train[1, ], type = "shap")
plot(ridge_breakdown)

# Figure 2 Panel A
# Now let's talk about global model explanation
library(DALEX)
library(ggplot2)
library(dplyr)
library(forcats)

set.seed(1234)
vip_xgb <- model_parts(
  explainer_xgb,
  loss_function = loss_root_mean_square,
  type = "difference"  # default: measures drop in performance
)

ggplot_imp <- function(vip_result) {
  metric_name <- attr(vip_result, "loss_name")
  metric_lab <- paste0(
    metric_name,
    " after permutations\n(higher = more important)"
  )
  
  baseline <- vip_result %>%
    filter(variable == "_full_model_") %>%
    summarise(dropout_loss = mean(dropout_loss)) %>%
    pull(dropout_loss)
  
  vip_clean <- vip_result %>%
    filter(!(variable %in% c("_full_model_", "_baseline_"))) %>%
    mutate(variable = fct_reorder(variable, dropout_loss))
  
  ggplot(vip_clean, aes(x = dropout_loss, y = variable)) +
    geom_vline(xintercept = baseline, linetype = "dashed", color = "gray40", linewidth = 1) +
    geom_boxplot(fill = "#1f77b4", alpha = 0.3, color = "#1f77b4") +
    labs(
      title = "Global Variable Importance - XGBoost",
      x = metric_lab,
      y = NULL
    ) +
    theme_minimal()
}

ggplot_imp(vip_xgb)

# Now, let's try another global explication method.
set.seed(1234)


# PDP plots
ggplot_pdp <- function(obj, x) {
  p <- 
    as_tibble(obj$agr_profiles) %>%
    mutate(`_label_` = stringr::str_remove(`_label_`, "^[^_]*_")) %>%
    ggplot(aes(x = `_x_`, y = `_yhat_`)) +
    geom_line(data = as_tibble(obj$cp_profiles),
              aes(x = .data[[x]], group = `_ids_`),
              linewidth = 0.5, alpha = 0.05, color = "gray50")
  
  num_colors <- n_distinct(obj$agr_profiles$`_label_`)
  
  if (num_colors > 1) {
    p <- p + geom_line(aes(color = `_label_`), linewidth = 1.2, alpha = 0.8)
  } else {
    p <- p + geom_line(color = "midnightblue", linewidth = 1.2, alpha = 0.8)
  }
  
  return(p)
}


# Generate PDP for EARN_change
pdp_EARN <- model_profile(explainer_xgb, N = 1000, variables = "EARN")

# Figure 3
# Plot PDP
ggplot_pdp(pdp_EARN, "EARN") +
  labs(
    title = "Partial Dependence Plot - EARN",
    x = "EARN",
    y = "Predicted EARN_lead_1",
    color = NULL
  ) +
  theme_minimal()

# Figure 4
# Generate PDP for efftax
pdp_efftax <- model_profile(explainer_xgb, N = 1000, variables = "efftax")

# Plot PDP
ggplot_pdp(pdp_efftax, "efftax") +
  labs(
    title = "Partial Dependence Plot - efftax",
    x = "efftax",
    y = "Predicted EARN_lead_1",
    color = NULL
  ) +
  theme_minimal()
# Figure 5
# Generate PDP for dpr
pdp_dpr <- model_profile(explainer_xgb, N = 1000, variables = "dpr")

# Plot PDP
ggplot_pdp(pdp_dpr, "dpr") +
  labs(
    title = "Partial Dependence Plot - dpr",
    x = "dpr",
    y = "Predicted EARN_lead_1",
    color = NULL
  ) +
  theme_minimal()

# PDP plots using patchwork
library(patchwork)

set.seed(1805)
# Create partial dependence plots for EARN, MVE, and ATO
pdp_earn <- model_profile(explainer_xgb, N = 5000, variables = "EARN")


p1 <- ggplot_pdp(pdp_earn, "EARN") + 
  labs(title = "PDP - EARN", x = "EARN", y = "Predicted Earnings")


p1 

# Load library
library(xgboost)

# Extract the fitted xgboost model from the tidymodels workflow
xgb_booster <- extract_fit_parsnip(xgb_fit)$fit

# Convert predictors to matrix (required by shapviz)
X_matrix <- vip_train |> as.matrix()

# Compute SHAP values
library(shapviz)
sv_xgb <- shapviz(xgb_booster, X_pred = X_matrix, approxcontrib = TRUE)

library(ggplot2)
library(shapviz)

# Feature importance summary bar plot (SHAP values)
sv_importance(sv_xgb) +
  ggtitle("SHAP Feature Importance - XGBoost")

# For Earn
sv_dependence(sv_xgb, v = "EARN") +
  ggtitle("SHAP Dependence Plot - EARN")



# Global Main and Interaction Effects (via hstats)
library(hstats)
library(ggplot2)
library(dplyr)
library(tidyr)

# Compute hstats object
hstats_obj <- hstats(
  object = explainer_xgb,  # DALEX explainer object
  type = "shap",
  X = vip_train             # Your predictor data
)

# Model performance
mp <- model_performance(explainer_xgb)
plot(mp)

# Ceteris Paribus profile for the first observation
cp_profile <- predict_profile(explainer_xgb, new_observation = vip_train[1, ])
plot(cp_profile)

# Figure 2 Panel B
#Load library
library(hstats)

# Run hstats on the explainer
# NOTE: this did not work when I used a workflow instead of the extracted model
# to define the explainer
#this takes a while to run but gives a progress bar
s <- hstats(
  object = explainer_xgb,
  approx = TRUE,
  n_max = 10000
)

#plot H-stats and interactions
plot(s)

#plot raw absolute H stats
plot(h2_pairwise(s, normalize = FALSE, squared = FALSE))

#pdps of one variable by buckets of the other
partial_dep(explainer_xgb, v = "EARN", BY = "dpr", type = "partial") |> 
  plot(show_points = FALSE)
partial_dep(explainer_xgb, v = "EARN", BY = "efftax", type = "partial") |> 
  plot(show_points = FALSE)
#google the difference between type partial and type accumulated
# you can use accumulated in the commands above too if desired
partial_dep(explainer_xgb, v = "EARN", BY = "efftax", type = "accumulated") |> 
  plot(show_points = FALSE)

partial_dep(explainer_xgb, v = "EARN", BY = "dpr", type = "accumulated") |> 
  plot(show_points = FALSE)

# Fogure 6 Panel A
# Strongest relative interaction (different visualizations)
ice(explainer_xgb, v = "EARN", BY = "dpr") |> 
  plot(center = TRUE)

# Fogure 7 Panel A
# Strongest relative interaction (different visualizations)
# now we can really see it working 
ice(explainer_xgb, v = "EARN", BY = "efftax") |> 
  plot(center = TRUE)

# # Fogure 6 Panel B
partial_dep(explainer_xgb, v = c("EARN", "dpr"), grid_size = 2000) |> 
  plot()
# Fogure 7 Panel B
partial_dep(explainer_xgb, v = c("EARN", "efftax"), grid_size = 2000) |> 
  plot()
pd_importance(s) |> 
  plot()


