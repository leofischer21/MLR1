library(caret)
library(ggplot2)

# Daten laden
data(iris)

# Train-Test-Split
set.seed(123)
trainIndex <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
train <- iris[trainIndex, ]
test <- iris[-trainIndex, ]

# Modell trainieren (Random Forest)
model <- train(Species ~ ., data = train, method = "rf",
               trControl = trainControl(method = "cv", number = 5),
               tuneLength = 3)

# Vorhersagen machen
predictions <- predict(model, newdata = test)

# Accuracy ausgeben
accuracy <- sum(predictions == test$Species) / nrow(test)
print(paste("Accuracy:", round(accuracy, 4)))

# Confusion Matrix
conf_mat <- confusionMatrix(predictions, test$Species)
print(conf_mat)

# Variable Importance plot
varImpPlot <- varImp(model)
plot(varImpPlot)