#--------------------------------------------------------------------------
#"""Special structure"
#--------------------------------------------------------------------------
#"""Model 1: predicting the probability of a claim"""
#--------------------------------------------------------------------------

df_model_fs <- df_model_selected %>%
  mutate(has_claim = ifelse(Cost_claims_year > 0, "1", "0")) %>%  # Note: "1"/"0" as strings!
  mutate(has_claim = factor(has_claim, levels = c("0", "1"))) %>%
  select(-Cost_claims_year)

# بررسی نوع دقیق متغیر
str(df_model_fs$has_claim)

# Convert all difftime columns to numeric
df_model_fs <- df_model_fs %>%
  mutate(across(where(~ inherits(.x, "difftime")), ~ as.numeric(.x, units = "days")))

# Convert all character columns to factor
df_model_fs$has_claim <- factor(df_model_fs$has_claim, levels = c(0, 1))


str(df_model_fs)
# Task classification
task_class <- TaskClassif$new(
  id = "has_claim_task",
  backend = df_model_fs,
  target = "has_claim",
  positive = "1"
)

# Preprocessing Pipeline 
pipeline_fs <- po("imputemean", affect_columns = selector_type("numeric")) %>>%
  po("imputemode", affect_columns = selector_type("factor")) %>>%
  po("encode", method = "one-hot")

# implementing pipeline
task_class_prep <- pipeline_fs$train(task_class)[[1]]


learner_class <- lrn("classif.xgboost", predict_type = "prob", objective = "binary:logistic")
resampling_class <- rsmp("cv", folds = 5)

rr_class <- resample(task_class_prep, learner_class, resampling_class, store_models = TRUE)
rr_class$aggregate(measures = list(msr("classif.auc"), msr("classif.acc")))


learner_class$train(task_class_prep)
pred_class <- learner_class$predict(task_class_prep)

# saving the probability of claim occurrence for each row
df_model_fs$prob_claim <- pred_class$prob[, "1"]


# ---------------------------------------------------------------------------
#"""Model 2: Frequency of a claim"""
# ---------------------------------------------------------------------------
df_freq <- df_model_selected %>%
  filter(Cost_claims_year > 0) %>%
  select(-Cost_claims_year)  

# Checking N_claims_year == 0
table(df_freq$N_claims_year == 0)

# definging task frequency
task_freq <- TaskRegr$new(
  id = "claims_frequency",
  backend = df_freq,
  target = "N_claims_year"
)

# Pipeline like before
task_freq_prep <- pipeline_fs$train(task_freq)[[1]]

# ---- Define Learner ----
learner_freq <- lrn("regr.ranger", id = "RF_frequency", importance = "impurity")
resampling_freq <- rsmp("cv", folds = 5)

rr_freq <- resample(task_freq_prep, learner_freq, resampling_freq, store_models = TRUE)
rr_freq$aggregate(measures = list(msr("regr.rmse"), msr("regr.rsq")))


learner_freq$train(task_freq_prep)
pred_freq <- learner_freq$predict(task_freq_prep)

#  saving the frequency of claim occurrence for each row
df_freq$predicted_freq <- pred_freq$response



#----------------------------------------------------------------------------
#"""Model 3: Severity of a claim"""
#----------------------------------------------------------------------------
df_sev <- df_model_selected %>%
  filter(Cost_claims_year > 0) %>%
  mutate(log_cost = log1p(Cost_claims_year))  # log(1 + x)

task_sev <- TaskRegr$new(
  id = "claims_severity",
  backend = df_sev,
  target = "log_cost"
)

task_sev_prep <- pipeline_fs$train(task_sev)[[1]]

learner_sev <- lrn("regr.xgboost",
                   objective = "reg:squarederror",
                   nrounds = 300,
                   eta = 0.05,
                   max_depth = 6,
                   subsample = 0.8,
                   colsample_bytree = 0.8
)

rr_sev <- resample(task_sev_prep, learner_sev, rsmp("cv", folds = 5), store_models = TRUE)
rr_sev$aggregate(measures = list(msr("regr.rmse"), msr("regr.rsq")))



learner_sev$train(task_sev_prep)
pred_sev <- learner_sev$predict(task_sev_prep)

df_sev$predicted_severity <- expm1(pred_sev$response)  # معکوس log1p




# ---------------------------------------------------------------------------
#"""Final Model"""
# ---------------------------------------------------------------------------
# 1. Add ID columns to each dataframe
df_model$ID <- 1:nrow(df_model)
df_freq$ID <- rownames(df_freq) |> as.integer()
df_sev$ID <- rownames(df_sev) |> as.integer()
df_model_fs$ID <- 1:nrow(df_model_fs)

# 2. Merge dataframes
df_merged <- df_model_fs %>%
  left_join(df_freq %>% select(ID, predicted_freq), by = "ID") %>%
  left_join(df_sev %>% select(ID, predicted_severity), by = "ID") %>%
  mutate(
    predicted_freq = coalesce(predicted_freq, 0),
    predicted_severity = coalesce(predicted_severity, 0),
    pred_total_cost = prob_claim * predicted_freq * predicted_severity
  )

df_compare <- df_model %>%
  select(ID, Cost_claims_year) %>%
  left_join(df_merged %>% select(ID, pred_total_cost), by = "ID")

# حالا روی کل دیتاست محاسبه کن
correlation <- cor(df_compare$Cost_claims_year, df_compare$pred_total_cost)
rmse_final <- sqrt(mean((df_compare$Cost_claims_year - df_compare$pred_total_cost)^2))

cat("Correlation:", round(correlation, 4), "\n")
cat("RMSE Frequency–Severity Model:", round(rmse_final, 2), "\n")


baseline_value <- mean(df_model$Cost_claims_year)
rmse_baseline <- sqrt(mean((df_model$Cost_claims_year - baseline_value)^2))

cat("RMSE Baseline Model (mean):", round(rmse_baseline, 2), "\n")


ggplot(df_compare, aes(x = Cost_claims_year, y = pred_total_cost)) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Predicted vs Actual Cost_claims_year (3-stage model)",
       x = "Actual Cost",
       y = "Predicted Cost") +
  theme_minimal()


baseline_value <- mean(df_compare$Cost_claims_year)
rmse_baseline <- sqrt(mean((df_compare$Cost_claims_year - baseline_value)^2))

cat("📌 RMSE Frequency–Severity Model:", round(rmse_final, 2), "\n")
cat("📌 RMSE Baseline Model (mean):", round(rmse_baseline, 2), "\n")



# --------------------------------------------------------------------------
"""# Log Severity model"""
# --------------------------------------------------------------------------
df_sev_log <- df_model %>%
  filter(Cost_claims_year > 0) %>%
  mutate(log_severity = log1p(Cost_claims_year))

task_sev_log <- TaskRegr$new(
  id = "claims_severity_log",
  backend = df_sev_log,
  target = "log_severity"
)

task_sev_log_prep <- pipeline_fs$train(task_sev_log)[[1]]

learner_sev_log <- lrn("regr.xgboost", objective = "reg:squarederror", nrounds = 100)
resampling_sev_log <- rsmp("cv", folds = 5)

rr_sev_log <- resample(task_sev_log_prep, learner_sev_log, resampling_sev_log)
rr_sev_log$aggregate(measures = list(msr("regr.rmse"), msr("regr.rsq")))

learner_sev_log$train(task_sev_log_prep)
pred_sev_log <- learner_sev_log$predict(task_sev_log_prep)

# بازگردانی از log به مقدار واقعی
df_sev_log$predicted_severity_log <- expm1(pred_sev_log$response)


df_sev_log$ID <- rownames(df_sev_log) |> as.integer()

df_merged_log <- df_model_fs %>%
  left_join(df_freq %>% select(ID, predicted_freq), by = "ID") %>%
  left_join(df_sev_log %>% select(ID, predicted_severity_log), by = "ID") %>%
  mutate(
    pred_total_cost_log = prob_claim * predicted_freq * predicted_severity_log
  )

df_compare_log <- df_model %>%
  select(ID, Cost_claims_year) %>%
  left_join(df_merged_log %>% select(ID, pred_total_cost_log), by = "ID") %>%
  drop_na()

# محاسبه RMSE
rmse_final_log <- sqrt(mean((df_compare_log$Cost_claims_year - df_compare_log$pred_total_cost_log)^2))
rmse_baseline_log <- sqrt(mean((df_compare_log$Cost_claims_year - mean(df_compare_log$Cost_claims_year))^2))

cat("📌 RMSE Frequency–Severity with log severity:", round(rmse_final_log, 2), "\n")
cat("📌 RMSE Baseline (mean):", round(rmse_baseline_log, 2), "\n")