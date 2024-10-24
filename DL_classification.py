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


## xgboost
from xgboost import XGBClassifier
from xgboost import plot_importance


print("\nXGBoostClassifier\n")
xgb_cl = XGBClassifier(
    n_estimators=100,          # Number of trees
    max_depth=9,               # Maximum depth of each tree
    learning_rate=0.05,        # Step size shrinkage
    subsample=0.8,             # Fraction of samples used for training
    colsample_bytree=0.8,      # Fraction of features used for each tree
    gamma=0.1,                 # Minimum loss reduction
    reg_alpha=0.01,            # L1 regularization (Lasso)
    reg_lambda=1.5,            # L2 regularization (Ridge)
    eval_metric='mlogloss'     # Evaluation metric
)
# Train the model
xgb_cl.fit(X_train, y_train)
xgb_predictions = xgb_cl.predict(X_test)
accuracy = accuracy_score(y_test, xgb_predictions)
randomForest_cm = confusion_matrix(y_test, xgb_predictions)
print(randomForest_cm)
print("Accuracy:", accuracy)

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],        # Number of trees
    'max_depth': [3, 6, 9],                # Maximum depth of each tree
    'learning_rate': [0.01, 0.05, 0.1],    # Step size shrinkage
    'subsample': [0.6, 0.8, 1.0],          # Fraction of samples to be used for training
    'colsample_bytree': [0.6, 0.8, 1.0],   # Fraction of features to be used for each tree
    'gamma': [0, 0.1, 0.3],                # Minimum loss reduction required for further partitioning
    'reg_alpha': [0, 0.01, 0.1],           # L1 regularization (Lasso)
    'reg_lambda': [1, 1.5, 2],             # L2 regularization (Ridge)
}

# Create a random forest classifier
xgb_cl = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Use random search to find the best hyperparameters
random_search = RandomizedSearchCV(
    estimator=xgb_cl, 
    param_distributions=param_grid, 
    n_iter=20,          # Number of random combinations to try
    scoring='accuracy', # Evaluation metric
    cv=5,               # 5-fold cross-validation
    verbose=1,          # Show progress
    random_state=42,    # For reproducibility
    n_jobs=-1           # Use all available cores
)
# Fit the random search object to the data
rand_search.fit(X_train, y_train)

# Create a variable for the best model
best_xgb_cl = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

# Generate predictions with the best model
best_xgb_cl_predictions = best_xgb_cl.predict(X_test)

# Create the confusion matrix
XGBoost_cm = confusion_matrix(y_test, best_xgb_cl_predictions)
accuracy = accuracy_score(y_test, best_xgb_cl_predictions)
precision = precision_score(y_test, best_xgb_cl_predictions, average='macro')
recall = recall_score(y_test, best_xgb_cl_predictions, average='macro')
f1 = f1_score(y_test, best_xgb_cl_predictions, average='macro')

print(XGBoost_cm)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1) # Calculate F1-score

from sklearn.metrics import classification_report
# Generate classification report
report = classification_report(y_test, best_randomForest_predictions)
print(report)


ConfusionMatrixDisplay(confusion_matrix=XGBoost_cm).plot()
plt.title('Confusion Matrix best RandomForest')
plt.show()
