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

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Dataset, y, test_size=0.33, random_state=1)

from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train,y_train)
svm_predictions = svm.predict(X_test)
svm_cm = confusion_matrix(y_test, svm_predictions)
print(svm_cm)
disp = ConfusionMatrixDisplay(confusion_matrix=svm_cm)
disp.plot()
plt.title('Confusion Matrix')
plt.show()

from sklearn.tree import DecisionTreeClassifier
tree_model= DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=1)
tree_model.fit(X_train,y_train)
tree_predictions = tree_model.predict(X_test)
tree_cm = confusion_matrix(y_test, tree_predictions)
print(tree_cm)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2',C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
lr.fit(X_train,y_train)
lr_predictions = lr.predict(X_test)
lr_cm = confusion_matrix(y_test, lr_predictions)
print(lr_cm)


# X_combined = np.vstack((X_train, X_test))
# y_combined = np.vstack((y_train, y_test))

## NN

# import tensorflow as tf
# from tensorflow import keras
# from keras import Model
# from keras.models import Sequential
# from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import Reshape
# from tensorflow.keras.layers import Conv2DTranspose
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.optimizers import Adam, SGD

# object for callback function during training
#loss_callback = LossCallback()

# _l1 = 32
# _activation = 'relu'
# _alpha = 0.05
# # define model
# inputs = Input(shape=(len(Data[0]),))
# hidden1 = Dense(_l1, activation=_activation)(inputs)
# hidden2 = Dense(_l1, activation=_activation)(hidden1)
# predictions = Dense(len(Data[0]), activation="softmax")(hidden2)

# model = Model(inputs=inputs, outputs=predictions)
# model.summary()

# # compile model with mse loss and ADAM optimizer (uncomment for SGD)
# model.compile(loss='mse', optimizer=Adam(learning_rate=_alpha))
# #history = model.fit(x=x_train, y=x_train, epochs=self._steps, verbose=0, batch_size=len(x_train), callbacks=[cp_callback, loss_callback])