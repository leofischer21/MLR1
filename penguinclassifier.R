# Penguins ML Showcase (R)
# (klassifiziert Pinguin-Arten mit hübschen Visuals)
# -------------------------------------------------
# Was das Script macht:
# - EDA mit Pair-Plot
# - ML-Pipeline (tidymodels) mit Random Forest + Cross-Validation
# - Confusion-Matrix als Heatmap
# - Feature Importance (VIP)
# - UMAP-Embedding (2D) für "wow"-Clusterbild
# - Speichert alle Plots in ./figs/

set.seed(42)

# -----------------------
# 0) Packages laden
# -----------------------
need <- c(
  "tidyverse", "tidymodels", "palmerpenguins", "GGally", "vip", "uwot", "ggrepel"
)

installed <- rownames(installed.packages())
for (p in need) {
  if (!p %in% installed) install.packages(p, dependencies = TRUE)
}

library(tidyverse)
library(tidymodels)
library(palmerpenguins)
library(GGally)
library(vip)
library(uwot)
library(ggrepel)

theme_set(theme_minimal(base_size = 14))

# Ordner für Plots
if (!dir.exists("figs")) dir.create("figs")

# -----------------------
# 1) Daten vorbereiten
# -----------------------
df <- penguins %>% 
  select(species, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, island, sex)

# Split Train/Test (stratifiziert nach species)
split <- initial_split(df, prop = 0.8, strata = species)
train <- training(split)
test  <- testing(split)

# Rezept: Imputation + Scaling + Dummies
rec <- recipe(species ~ ., data = train) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())

# -----------------------
# 2) EDA – Pair-Plot (nur numerische Features)
# -----------------------
num_cols <- c("bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g")
pairs_plot <- train %>%
  select(all_of(num_cols), species) %>%
  GGally::ggpairs(aes(color = species), columns = 1:4, progress = FALSE) +
  theme(legend.position = "bottom")

ggsave("figs/01_pairs.png", pairs_plot, width = 10, height = 8, dpi = 160)

# -----------------------
# 3) Modell: Random Forest + CV-Tuning
# -----------------------
rf_spec <- rand_forest(mtry = tune(), min_n = tune(), trees = 500) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(rf_spec)

set.seed(42)
cv <- vfold_cv(train, v = 5, strata = species)

# Hyperparameter grid
grid <- grid_regular(
  mtry(range = c(1L, length(num_cols) + length(unique(train$island)) + length(unique(train$sex)))),
  min_n(range = c(2L, 20L)),
  levels = 5
)

metric_set <- metric_set(accuracy, kap)

set.seed(42)
res <- tune_grid(
  wf, resamples = cv, grid = grid, metrics = metric_set, control = control_grid(save_pred = TRUE)
)

best <- select_best(res, "accuracy")
final_wf <- finalize_workflow(wf, best)

final_fit <- fit(final_wf, data = train)

# -----------------------
# 4) Evaluation auf Testdaten
# -----------------------
preds <- predict(final_fit, test) %>% bind_cols(predict(final_fit, test, type = "prob")) %>% bind_cols(test)

acc <- accuracy(preds, truth = species, estimate = .pred_class)
kap <- kap(preds, truth = species, estimate = .pred_class)
print(acc)
print(kap)

# Confusion Matrix Heatmap
cm <- conf_mat(preds, truth = species, estimate = .pred_class)
cm_plot <- autoplot(cm, type = "heatmap") +
  scale_fill_gradient(low = "#e0ecf4", high = "#08519c") +
  labs(title = "Confusion Matrix (Test)")

ggsave("figs/02_confusion_heatmap.png", cm_plot, width = 7, height = 6, dpi = 160)

# -----------------------
# 5) Feature Importance (VIP)
# -----------------------
rf_fit <- extract_fit_parsnip(final_fit)$fit
vip_plot <- vip(rf_fit, num_features = 10) + labs(title = "Feature Importance (Random Forest)")

ggsave("figs/03_vip.png", vip_plot, width = 7, height = 5, dpi = 160)

# -----------------------
# 6) UMAP-Embedding (2D)
# -----------------------
rec_prep <- prep(rec)

X_all <- bake(rec_prep, new_data = df) %>% select(-species)
y_all <- df$species

set.seed(42)
umap_emb <- uwot::umap(as.matrix(X_all), n_neighbors = 15, min_dist = 0.1, metric = "euclidean")

emb_df <- as_tibble(umap_emb) %>% setNames(c("UMAP1", "UMAP2")) %>% mutate(species = y_all)

umap_plot <- ggplot(emb_df, aes(UMAP1, UMAP2, color = species)) +
  geom_point(alpha = 0.9, size = 2) +
  stat_ellipse(level = 0.9, linewidth = 0.6, alpha = 0.2) +
  labs(title = "UMAP: Penguins Feature Space", subtitle = "Schöne Cluster nach Art") +
  theme(legend.position = "bottom")

ggsave("figs/04_umap.png", umap_plot, width = 8, height = 6, dpi = 180)

# -----------------------
# 7) Kleiner Model-Report in die Konsole
# -----------------------
cat("\n================ Model Report ================\n")
cat("Beste Hyperparameter (CV):\n"); print(best)
cat("\nTest Accuracy: ", round(acc$.estimate, 3), " | Cohen's Kappa: ", round(kap$.estimate, 3), "\n", sep = "")
cat("Plots gespeichert in ./figs/:\n - 01_pairs.png\n - 02_confusion_heatmap.png\n - 03_vip.png\n - 04_umap.png\n")

# Ende
