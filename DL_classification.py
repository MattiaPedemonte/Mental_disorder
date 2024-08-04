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

import pandas as pd
# Read the CSV file
Data = pd.read_csv("Dataset-Mental-Disorders-Clean.csv")
features = list(Data.columns)
print(features)

patient_number = Data.iloc[:,0]
Dataset = Data.iloc[:,1:-1]
print(list(Dataset.columns))
y = Data.iloc[:,-1]
print(y.name)

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


# Standardizza i dati
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(Dataset)

# Esegui PCA
pca = PCA(n_components=2)  # Cambia n_components se desideri più componenti
principal_components = pca.fit_transform(scaled_data)

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