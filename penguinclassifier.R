# Penguins ML Showcase - Minimal Version
set.seed(42)

# Load only essential packages
library(tidyverse)
library(tidymodels)
library(palmerpenguins)

# Create output directory
if (!dir.exists("figs")) dir.create("figs")

# Prepare data (remove missing values)
penguins_clean <- penguins %>% 
  select(species, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g) %>%
  drop_na()

# Train/test split
split <- initial_split(penguins_clean, prop = 0.8, strata = species)
train <- training(split)
test <- testing(split)

# Simple recipe
rec <- recipe(species ~ ., data = train) %>%
  step_normalize(all_numeric_predictors())

# Random Forest with importance enabled
rf_spec <- rand_forest(trees = 100) %>%
  set_engine("ranger", importance = "permutation") %>%
  set_mode("classification")

model <- workflow() %>%
  add_recipe(rec) %>%
  add_model(rf_spec) %>%
  fit(data = train)

# Basic evaluation
preds <- predict(model, test) %>% bind_cols(test)
accuracy <- accuracy(preds, truth = species, estimate = .pred_class)

# Simple confusion matrix
conf_mat <- preds %>% 
  conf_mat(truth = species, estimate = .pred_class)

# Basic plot
conf_plot <- autoplot(conf_mat, type = "heatmap") +
  ggtitle(paste("Accuracy:", round(accuracy$.estimate, 3)))

ggsave("figs/confusion_matrix.png", conf_plot, width = 6, height = 5)

# Simple scatter plot instead of VIP
scatter_plot <- train %>%
  ggplot(aes(x = bill_length_mm, y = bill_depth_mm, color = species)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "Penguin Species by Bill Dimensions") +
  theme_minimal()

ggsave("figs/scatter_plot.png", scatter_plot, width = 6, height = 4)

cat("Model accuracy:", round(accuracy$.estimate, 3), "\n")
cat("Plots saved to figs/ directory\n")