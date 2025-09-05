# 1. Pakete laden
library(caret)
library(ggplot2)
library(pROC)
library(dplyr)
library(reshape2)

# 2. Titanic-Datensatz laden
# (caret bringt einen Basis Titanic-Datensatz mit, sonst hier aus 'titanic' Package laden)
if (!require(titanic)) install.packages("titanic")
library(titanic)
data("titanic_train")

# 3. Daten vorbereiten
titanic <- titanic_train %>%
  select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare) %>%
  mutate(
    Survived = factor(Survived, labels = c("No", "Yes")),
    Pclass = factor(Pclass),
    Sex = factor(Sex)
  ) %>%
  filter(!is.na(Age))  # Einfachheit halber NA in Age raus

# 4. Daten splitten (Training / Test)
set.seed(123)
train_index <- createDataPartition(titanic$Survived, p = 0.8, list = FALSE)
train_data <- titanic[train_index, ]
test_data <- titanic[-train_index, ]

# 5. Modell trainieren - Random Forest
model_rf <- train(
  Survived ~ .,
  data = train_data,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary),
  metric = "ROC"
)

# 6. Vorhersage auf Testdaten
pred_probs <- predict(model_rf, newdata = test_data, type = "prob")
pred_class <- predict(model_rf, newdata = test_data)

# 7. ROC Kurve plotten
roc_obj <- roc(response = test_data$Survived, predictor = pred_probs$Yes)
plot(roc_obj, col = "blue", main = "ROC-Kurve für Titanic Überlebensvorhersage")

# 8. Confusion Matrix erstellen
conf_mat <- confusionMatrix(pred_class, test_data$Survived)
print(conf_mat)

# 9. Feature Importance plotten
imp <- varImp(model_rf)
imp_df <- data.frame(Feature = rownames(imp$importance), Importance = imp$importance$Overall)
ggplot(imp_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  ggtitle("Feature Importance im Random Forest Modell") +
  xlab("") +
  ylab("Wichtigkeit")

# 10. Überlebensrate nach Geschlecht und Klasse visualisieren
ggplot(titanic, aes(x = Pclass, fill = Survived)) +
  geom_bar(position = "fill") +
  facet_wrap(~Sex) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Überlebensrate nach Passagierklasse und Geschlecht", y = "Prozentualer Anteil", x = "Passagierklasse") +
  theme_minimal()
