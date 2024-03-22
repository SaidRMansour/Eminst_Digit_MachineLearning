"""
    Loading and preparation of data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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


"""
    Checking for NaN-values
"""

nan_in_train = train_df.isna().sum().sum()
print(f"Antal NaN værdier i træningsdata: {nan_in_train}")

nan_in_test = test_df.isna().sum().sum()
print(f"Antal NaN værdier i testdata: {nan_in_test}")

"""
    Selection, training and fine-tuning of a model
"""

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(max_depth=40, n_estimators=500, n_jobs=-1, random_state=42, verbose=2)

rfc.fit(X_train, y_train)


"""
    Fine-tuning - Gridsearchcv
"""

# Sætter parameter som bruges til fine-tuning
#params = {'max_depth': [40, 70],'n_estimators': [600, 1000]}

#grid_search_cv = GridSearchCV(RandomForestClassifier(random_state=42), params, n_jobs=-1, cv=3, verbose=2)

#grid_search_cv.fit(X_train, y_train)

# Printer de bedste parametre og nøjagtigheden for den bedste model
#print("Bedste parametre fundet:", grid_search_cv.best_params_)
#print("Bedste nøjagtighed opnået:", grid_search_cv.best_score_)

# Viser bedste estimat
#grid_search_cv.best_estimator_

#y_pred = grid_search_cv.predict(X_test)


"""
    Evaluation of the model
"""

from sklearn.metrics import accuracy_score

# Laver forudsigelser på testdata og udregner nøjagtigheden
y_pred = rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Nøjagtighed af modellen: {accuracy}")

# Evalueringen af max_depth: 40, n-estimators: 500 --> Accur: 0.9825 -> elapsed: 1.5 min finished -> Random forest (without fine-tuning)

"""
GRIDSEARCHCV Result:

Bedste parametre fundet: {'max_depth': 70, 'n_estimators': 1000}
Bedste nøjagtighed opnået: 0.9807957522385697
RandomForestClassifier(max_depth=70, n_estimators=1000, random_state=42,
                       verbose=2)

Nøjagtighed af modellen: 0.9828745718642966

"""
