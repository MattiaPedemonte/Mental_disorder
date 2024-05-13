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
Data = pd.read_csv("Dataset-Mental-Disorders.csv")
patient_number = Data.iloc[1,:]
unique_sadness_value = Data.Sadness.unique()
column_count = Data.Sadness.value_counts().values
print(column_count)
features = list(Data.columns)
print(features)

dummies = pd.get_dummies(Data, columns = ['Sadness'], dtype=int).values
print(dummies)
features = list(Data.columns)
print(features)
# print(Data)
# print(patient_number)
# View the first 5 rows
Data.head()


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