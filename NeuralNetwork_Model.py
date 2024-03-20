"""
    Loading and preparation of data
"""
import matplotlib as matplotlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Indlæser data fra CSV-filer
train_data_path = 'emnist-digits-train.csv'  # Fil med træningsdata
test_data_path = 'emnist-digits-test.csv'    # Fil med testdata

# Læser CSV-filer til pandas DataFrames
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

# Konverterer DataFrames til numpy arrays for lettere håndtering
train_data = train_df.to_numpy()
test_data = test_df.to_numpy()

# Splitter data op i features (X) og labels (y), og normaliserer pixelværdierne
X_train = train_data[:, 1:] / 255.0
y_train = train_data[:, 0]

X_test = test_data[:, 1:] / 255.0
y_test = test_data[:, 0]

# Viser dimensionerne
print(f"X_train shape: {X_train.shape}") # 239999 pic, 784 pixels (28x28) -> features
print(f"X_test shape: {X_test.shape}") # 39999 pic, 784 pixels (28x28) -> features

# Opdeler det fulde træningsæt i et mindre træningsæt og valideringssæt
#Bruger 5000 som valideringssæt og resten som træningsæt
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

"""
    Selection
    Create a model using the Sequential API
"""
model = keras.models.Sequential()


# Øger antallet af neuroner og tilføjer flere skjulte lag med dropout
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(200, activation="relu"))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dropout(0.3))

# Output layer
model.add(keras.layers.Dense(10, activation="softmax"))


"""
    Compile the model
"""
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

"""
    Train the model
"""

history = model.fit(X_train, y_train, epochs=50, validation_data=(X_valid, y_valid))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

"""
    Evaluate the model.
"""

evaluating = model.evaluate(X_test, y_test)
print(f"Model evalutating: {evaluating}")

# Lav forudsigelser uden sandsynligheder.
X_new = X_test[:3]
y_pred = model.predict(X_new)
classes=np.argmax(y_pred,axis=1)
print(classes)

# Tjek om forudsigelserne var korrekte.
y_new = y_test[:3]
print(y_new)

"""
     Results
"""

"""
Epoch 1/50
7344/7344 ━━━━━━━━━━━━━━━━━━━━ 7s 939us/step - accuracy: 0.7604 - loss: 0.7478 - val_accuracy: 0.9604 - val_loss: 0.1438
Epoch 2/50
7344/7344 ━━━━━━━━━━━━━━━━━━━━ 7s 922us/step - accuracy: 0.9449 - loss: 0.1937 - val_accuracy: 0.9730 - val_loss: 0.1004
.
.
.
Epoch 42/50
7344/7344 ━━━━━━━━━━━━━━━━━━━━ 7s 947us/step - accuracy: 0.9917 - loss: 0.0290 - val_accuracy: 0.9908 - val_loss: 0.0388
Epoch 43/50
7344/7344 ━━━━━━━━━━━━━━━━━━━━ 7s 942us/step - accuracy: 0.9918 - loss: 0.0284 - val_accuracy: 0.9906 - val_loss: 0.0392
Epoch 44/50
7344/7344 ━━━━━━━━━━━━━━━━━━━━ 7s 942us/step - accuracy: 0.9910 - loss: 0.0291 - val_accuracy: 0.9908 - val_loss: 0.0407
Epoch 45/50
7344/7344 ━━━━━━━━━━━━━━━━━━━━ 7s 950us/step - accuracy: 0.9913 - loss: 0.0287 - val_accuracy: 0.9908 - val_loss: 0.0396
Epoch 46/50
7344/7344 ━━━━━━━━━━━━━━━━━━━━ 7s 950us/step - accuracy: 0.9919 - loss: 0.0269 - val_accuracy: 0.9918 - val_loss: 0.0370
Epoch 47/50
7344/7344 ━━━━━━━━━━━━━━━━━━━━ 7s 947us/step - accuracy: 0.9921 - loss: 0.0262 - val_accuracy: 0.9916 - val_loss: 0.0363
Epoch 48/50
7344/7344 ━━━━━━━━━━━━━━━━━━━━ 7s 929us/step - accuracy: 0.9925 - loss: 0.0258 - val_accuracy: 0.9904 - val_loss: 0.0399
Epoch 49/50
7344/7344 ━━━━━━━━━━━━━━━━━━━━ 7s 935us/step - accuracy: 0.9918 - loss: 0.0264 - val_accuracy: 0.9904 - val_loss: 0.0379
Epoch 50/50
7344/7344 ━━━━━━━━━━━━━━━━━━━━ 7s 930us/step - accuracy: 0.9919 - loss: 0.0273 - val_accuracy: 0.9902 - val_loss: 0.0370


Model evalutating: [0.031229550018906593, 0.991199791431427]
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step
[9 7 9]
[9 7 9]

"""