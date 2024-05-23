#install.packages("caTools")
library(caTools)
library(mlr)

# Einlesen des Datensatzes
datensatz <- read.csv("heart_disease_health_indicators_BRFSS2015.csv", header=T)

# Zusammenfassung und Überblick über den Datensatz
summary(datensatz)
str(datensatz)
head(datensatz)
names(datensatz)

# Umwandeln der Zielvariable in einen Faktor
datensatz$HeartDiseaseorAttack <- as.factor(datensatz$HeartDiseaseorAttack)

# Festlegen des Seeds für reproduzierbare Ergebnisse
set.seed(123)

# Teilen der Daten in Training- und Testsets
split <- sample.split(datensatz$HeartDiseaseorAttack, SplitRatio = 0.7)
train_cl <- subset(datensatz, split==TRUE)
test_cl <- subset(datensatz, split==FALSE)

# Wir definieren zunächst die Resampling-Strategie
resampling_strategy <- makeResampleDesc("CV", iters = 5)

# Anwendung von Sequential Feature Selector (SFS)
task <- makeClassifTask(id = "mydata_subset", data = train_cl, target = "HeartDiseaseorAttack")
learner <- makeLearner("classif.logreg", predict.type = "prob")
ctrl <- makeFeatSelControlSequential(method = "sbs", maxit = NA)
# ...
if (!file.exists("selected_features.rds")) {
  # Führe den SBS aus und speichere die Features erneut, wenn die Datei nicht existiert
  res <- selectFeatures(learner, task, resampling = resampling_strategy, control = ctrl)
  selected_features <- res$x
  saveRDS(selected_features, "selected_features.rds")
} else {
  # Lädt die ausgewählten Merkmale, wenn die Datei existiert
  selected_features <- readRDS("selected_features.rds")
}

significant_features <- c('HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'Diabetes',
              'PhysActivity', 'HvyAlcoholConsump', 'NoDocbcCost', 'GenHlth', 'DiffWalk', 'Sex', 'Age', 'Income')
print(names(train_cl))  # Let's check the column names right before modeling
# Erstellen Sie das glm-Modell nur mit den ausgewählten Features.
formula <- as.formula(paste("HeartDiseaseorAttack ~ ", paste(significant_features, collapse = " + ")))
modell <- glm(formula, data = train_cl, family = binomial)

summary(modell)

# Berechnen der Wahrscheinlichkeiten auf dem Testset
probs <- predict(modell, newdata = test_cl[, significant_features], type = "response")

# Umwandeln der Wahrscheinlichkeiten in Klassenlabels
prediction <- ifelse(probs > 0.05, 1, 0)

# Berechnung der Konfusionsmatrix
confusionMatrix <- table(Predicted = prediction, Actual = test_cl$HeartDiseaseorAttack)
print(confusionMatrix)

TP <- confusionMatrix[2, 2]
TN <- confusionMatrix[1, 1]
FP <- confusionMatrix[2, 1]
FN <- confusionMatrix[1, 2]

# Berechnung von Recall (Sensitivity)
recall <- TP / (TP + FN)

# Berechnung von Specificity
specificity <- TN / (TN + FP)

# Berechnung der Genauigkeit
accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)

# Ausgabe von Recall und Specificity
print(paste('Sensitivitaet:', recall))
print(paste('Spezifitaet:', specificity))
print(paste('Genauigkeit:', accuracy))

coef(modell)
#boxplot(datensatz$BMI, datensatz$HeartDiseaseorAttack, main = "Alter vs. Cholesterin", xlab = "Alter", ylab = "Cholesterin")
#boxplot(datensatz$Age ~ datensatz$HeartDiseaseorAttack, main = "Risikofaktoren von Herzkrankheiten bzw. Infarkten", xlab = "Herzkrankheit oder Infarkt (0 = nein, 1 = ja)", ylab = "Alter")


#person_age <- 34
#breaks <- c(18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, Inf)
#age_class <- findInterval(person_age, breaks)
#print(paste("Altersklasse =", age_class))

# Erzeugen Sie einen neuen Datenrahmen, der die Werte des Einzelfalls darstellt
#einzelfall_person <- data.frame(
  #HighBP = 0,
  #HighChol = 0,
  #CholCheck = 0,
  #Smoker = 1,
  #Stroke = 0,
  #Diabetes = 0,
  #PhysActivity = 0,
  #HvyAlcoholConsump = 0,
  #NoDocbcCost = 0,
  #GenHlth = 3,
  #DiffWalk = 0,
  #Sex = 0,
  #Age = age_class, # 14 Alterskategorien: 18 Jahre = 1; 80 = 13
  #Income = 6
#)

# Behalten Sie nur die ausgewählten Merkmale für einzelfall_person
#einzelfall_person <- einzelfall_person[, names(einzelfall_person) %in% selected_features]

# Verwenden Sie das Modell, um die Vorhersage zu berechnen
#probs <- predict(modell, newdata = einzelfall_person, type="response")

# Wandeln Sie die Wahrscheinlichkeiten in Klassenlabels um
#prediction <- ifelse(probs > 0.5, 1, 0)

#percentage <- probs * 100
#print(prediction)
# Zeigen Sie die Vorhersage an für den Einzelfall
#print(paste("Wahrscheinlichkeit fuer Herzkrankheiten =", percentage, "%"))


# todo Zielkoeffizient ändern um neue Aussagen zu treffen z.B. Diabetes-Wahrscheinlichkeit
# todo weitere Machine Learning Verfahren ausprobieren um genauere Prognosen zu erstellen