import pandas as pd
# Read the CSV file
Data = pd.read_csv(".\Dataset-Mental-Disorders-Clean.csv")
features = list(Data.columns)
print(f"features: {features}")

patient_number = Data.iloc[:,0]
Dataset = Data.iloc[:,1:-1]
print(f"x: {list(Dataset.columns)}")
y = Data.iloc[:,-1]
print(f"y: {y.name}")

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


# Standardizza i dati
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#scaled_data = scaler.fit_transform(Dataset)

# Esegui PCA
pca = PCA(n_components=2)  # Cambia n_components se desideri più componenti
principal_components = pca.fit_transform(Dataset)

# Crea un DataFrame con i risultati PCA
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Unisci i risultati PCA con le etichette originali (se presenti)
final_df = pd.concat([pca_df, y], axis=1)  # Supponendo che ci sia una colonna 'label'

# Plot PCA
plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', hue='Expert Diagnose', data=final_df, palette='viridis')
plt.title('PCA di due componenti principali')
plt.xlabel('Prima componente principale')
plt.ylabel('Seconda componente principale')
plt.show()


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

X_train, X_test, y_train, y_test = train_test_split(Dataset, y, test_size=0.33, random_state=1)

## SVM
from sklearn.svm import SVC
print("\nSVM\n")
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train,y_train)
svm_predictions = svm.predict(X_test)
accuracy = accuracy_score(y_test, svm_predictions)
svm_cm = confusion_matrix(y_test, svm_predictions)
print(svm_cm)
print("Accuracy:", accuracy)

# show confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=svm_cm)
disp.plot()
plt.title('Confusion Matrix SVM')
plt.show()

 ## DecisionTree
from sklearn.tree import DecisionTreeClassifier
print("\nDecisionTreeClassifier\n")
tree_model= DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=1)
tree_model.fit(X_train,y_train)
tree_predictions = tree_model.predict(X_test)
accuracy = accuracy_score(y_test, tree_predictions)
tree_cm = confusion_matrix(y_test, tree_predictions)
print(tree_cm)
print("Accuracy:", accuracy)
## LogisticRegression
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(penalty='l2',C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
# lr.fit(X_train,y_train)
# lr_predictions = lr.predict(X_test)
# lr_cm = confusion_matrix(y_test, lr_predictions)
# print(lr_cm)


## RandomForest
from sklearn.ensemble import RandomForestClassifier
print("\nRandomForestClassifier\n")
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
randomForest_predictions = rf.predict(X_test)
accuracy = accuracy_score(y_test, randomForest_predictions)
randomForest_cm = confusion_matrix(y_test, randomForest_predictions)
print(randomForest_cm)
print("Accuracy:", accuracy)

param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

# Generate predictions with the best model
best_randomForest_predictions = best_rf.predict(X_test)

# Create the confusion matrix
randomForest_cm = confusion_matrix(y_test, best_randomForest_predictions)
accuracy = accuracy_score(y_test, best_randomForest_predictions)
precision = precision_score(y_test, best_randomForest_predictions, average='macro')
recall = recall_score(y_test, best_randomForest_predictions, average='macro')
f1 = f1_score(y_test, best_randomForest_predictions, average='macro')

print(randomForest_cm)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1) # Calculate F1-score

from sklearn.metrics import classification_report
# Generate classification report
report = classification_report(y_test, best_randomForest_predictions)
print(report)


ConfusionMatrixDisplay(confusion_matrix=randomForest_cm).plot()
plt.title('Confusion Matrix best RandomForest')
plt.show()
