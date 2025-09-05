data(iris)

# Nur Zahlen nehmen
iris_num <- iris[, 1:4]

# 3 Cluster erstellen
km <- kmeans(iris_num, centers = 3)

# Cluster-Zuordnung anschauen
table(km$cluster, iris$Species)