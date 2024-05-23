#install.packages("caTools")
#install.packages("randomForest")
#install.packages("mlr")
library(caTools)
library(randomForest)
library(mlr)

# Einlesen des Datensatzes
datensatz <- read.csv("heart_disease_health_indicators_BRFSS2015.csv", header=T)

# Zusammenfassung und Überblick über den Datensatz
summary(datensatz)
str(datensatz)
head(datensatz)

# Umwandeln der Zielvariable in einen Faktor
datensatz$HeartDiseaseorAttack <- as.factor(datensatz$HeartDiseaseorAttack)

# Festlegen des Seeds für reproduzierbare Ergebnisse
set.seed(123)

# Teilen der Daten in Training- und Testsets
split <- sample.split(datensatz$HeartDiseaseorAttack, SplitRatio = 0.7)
train_cl <- subset(datensatz, split==TRUE)
# Erstellt einen Datensatz nur mit den signifikanten Merkmalen
train_cl_subset <- train_cl[, c('HeartDiseaseorAttack', 'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'Diabetes',
             'PhysActivity', 'HvyAlcoholConsump', 'NoDocbcCost', 'GenHlth', 'DiffWalk', 'Sex', 'Age', 'Income')]
test_cl <- subset(datensatz, split==FALSE)



#res$measures.test


random_forest <- randomForest(HeartDiseaseorAttack ~ ., data = train_cl_subset, ntree = 500, mtry = 14, min.node.size = 2)

# Ausgabe der Modellzusammenfassung
print(random_forest)

# Vorhersagen auf dem Testset berechnen
predictions_forest_prob <- predict(random_forest, newdata = test_cl, type = "prob")

# Klassifizierungsschwelle anpassen
predictions_forest <- ifelse(predictions_forest_prob[,2] > 0.1, 1, 0)

# Ausgabe der Konfusionsmatrix
confusionMatrix_forest <- table(Predicted = predictions_forest, Actual = test_cl$HeartDiseaseorAttack)
print(confusionMatrix_forest)

# Berechnung von Recall (Sensitivity), Specificity und Accuracy
TP_forest <- confusionMatrix_forest[2, 2]
TN_forest <- confusionMatrix_forest[1, 1]
FP_forest <- confusionMatrix_forest[2, 1]
FN_forest <- confusionMatrix_forest[1, 2]

recall_forest <- TP_forest / (TP_forest + FN_forest)
specificity_forest <- TN_forest / (TN_forest + FP_forest)
accuracy_forest <- sum(diag(confusionMatrix_forest)) / sum(confusionMatrix_forest)

# Ausgabe von Recall, Specificity und Accuracy
print(paste('Sensitivity Random Forest:', recall_forest))
print(paste('Specificity Random Forest:', specificity_forest))
print(paste('Accuracy Random Forest:', accuracy_forest))

varImpPlot(random_forest, main="Feature Importance")