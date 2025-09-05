library(rpart)
library(rpart.plot)

# Modell bauen
tree <- rpart(Species ~ ., data = iris)

# Baum plotten
rpart.plot(tree)