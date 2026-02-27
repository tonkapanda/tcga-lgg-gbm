library(tidyverse)
library(tidymodels)


set.seed(1)

uci_data <- read_csv("data/TCGA_InfoWithGrade.csv") |>
  mutate(Grade = as.factor(Grade))

uci_split <- initial_split(uci_data, prop = 0.75, strata = Grade)
uci_train <- training(uci_split)
uci_test  <- testing(uci_split)

# 2. MODEL & RECIPE ================================================
knn_spec <- nearest_neighbor(
  neighbors = tune(), 
  weight_func = tune(), 
  dist_power = tune()
) |>
  set_engine("kknn") |>
  set_mode("classification")

# age normalization

uci_recipe <- recipe(Grade ~ ., data = uci_train) |>
  step_normalize(all_predictors())

uci_workflow <- workflow() |>
  add_recipe(uci_recipe) |>
  add_model(knn_spec)

cv_folds <- vfold_cv(uci_train, v = 5, strata = Grade)

knn_grid <- grid_regular(
  neighbors(range = c(1, 30)),
  weight_func(),
  dist_power(),
  levels = 15
)

tune_results <- uci_workflow |>
  tune_grid(
    resamples = cv_folds,
    grid = knn_grid,
    metrics = metric_set(accuracy, roc_auc, precision, recall, f_meas, mcc)
  )


best_knn <- select_best(tune_results, metric = "roc_auc")
final_wf <- finalize_workflow(uci_workflow, best_knn)
final_fit <- last_fit(final_wf, uci_split)


test_metrics <- collect_metrics(final_fit)
print("--- final metrics ---")
print(test_metrics)

cm_plot <- collect_predictions(final_fit) |>
  conf_mat(truth = Grade, estimate = .pred_class) |>
  autoplot(type = "heatmap") +
  labs(title = "k-NN Confusion Matrix: UCI Glioma Dataset",
       subtitle = "Predicting Grade (LGG vs. GBM)")

ggsave("plots/uci_benchmark_cm.png", plot = cm_plot, width = 6, height = 5)


uci_knn_detailed_plot <- tune_results |>
  collect_metrics() |>
  filter(.metric == "roc_auc") |>
  ggplot(aes(x = neighbors, y = mean, color = weight_func)) +
  geom_line(linewidth = 1) +
  geom_point() +
  facet_wrap(~ dist_power, labeller = label_both) +
  scale_color_viridis_d(option = "plasma") +
  labs(
    title = "k-NN Multi-Parameter Tuning: UCI Benchmark",
    subtitle = "Faceted by Distance Power (1 = Manhattan, 2 = Euclidean)",
    x = "Number of Neighbors (k)",
    y = "Mean ROC-AUC (5-Fold CV)",
    color = "Weight Function"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave("plots/uci_knn_detailed_tuning.png", plot = uci_knn_detailed_plot, width = 12, height = 8)