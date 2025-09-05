library(pROC)

# Iris auf 2 Klassen filtern
iris_binary <- subset(iris, Species != "setosa")
iris_binary$Species <- factor(iris_binary$Species)

set.seed(123)
trainIndex <- createDataPartition(iris_binary$Species, p = 0.7, list = FALSE)
train <- iris_binary[trainIndex, ]
test <- iris_binary[-trainIndex, ]

model <- train(Species ~ ., data = train, method = "rf",
               trControl = trainControl(method = "cv", number = 5,
                                        classProbs = TRUE, summaryFunction = twoClassSummary),
               metric = "ROC",
               tuneLength = 3)

# Vorhersage Wahrscheinlichkeit
pred_probs <- predict(model, test, type = "prob")

roc_obj <- roc(response = test$Species, predictor = pred_probs[,2])
plot(roc_obj, main = "ROC Kurve für Random Forest")