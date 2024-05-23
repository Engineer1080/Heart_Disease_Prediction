#install.packages("caTools")
#install.packages("rpart.plot")
library(caTools)
library(rpart)
library(rpart.plot)

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
test_cl <- subset(datensatz, split==FALSE)


# Entscheidungsbaum
# Anpassen des Entscheidungsbaum-Modells
decision_tree <- rpart(HeartDiseaseorAttack ~ HighBP + HighChol + CholCheck + Smoker + Stroke + Diabetes + PhysActivity + HvyAlcoholConsump + NoDocbcCost + GenHlth + DiffWalk + Sex + Age + Income, data = train_cl, method = "class", control = rpart.control(minsplit = 1, cp = 0.001))

# Ausgabe der Modellzusammenfassung
summary(decision_tree)

# Berechnen der Vorhersagen auf dem Testset
predictions_tree_prob <- predict(decision_tree, test_cl, type = "prob")

# Klassifizierungsschwelle anpassen
predictions_tree <- ifelse(predictions_tree_prob[,2] > 0.1, 1, 0)

# Ausgabe der Konfusionsmatrix
confusionMatrix_tree <- table(Predicted = predictions_tree, Actual = test_cl$HeartDiseaseorAttack)
print(confusionMatrix_tree)


TP_tree <- confusionMatrix_tree[2, 2]
TN_tree <- confusionMatrix_tree[1, 1]
FP_tree <- confusionMatrix_tree[2, 1]
FN_tree <- confusionMatrix_tree[1, 2]

# Berechnung von Recall (Sensitivity)
recall_tree <- TP_tree / (TP_tree + FN_tree)

# Berechnung von Specificity
specificity_tree <- TN_tree / (TN_tree + FP_tree)

# Berechnung der Genauigkeit
accuracy_tree <- sum(diag(confusionMatrix_tree)) / sum(confusionMatrix_tree)

# Ausgabe von Recall und Specificity
print(paste('Sensitivitaet Entscheidungsbaum:', recall_tree))
print(paste('Spezifitaet Entscheidungsbaum:', specificity_tree))
print(paste('Genauigkeit Entscheidungsbaum:', accuracy_tree))

# Plot des Entscheidungsbaums
rpart.plot(decision_tree, type = 5, extra = 101, under = TRUE, tweak = 1.6, box.palette="RdBu", main = "Decision Tree for Heart Disease Prediction")



# todo Zielkoeffizient ändern um neue Aussagen zu treffen z.B. Diabetes-Wahrscheinlichkeit