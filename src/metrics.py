import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from keras.models import load_model

# Load the processed data
data = pd.read_csv('../dataset/processed_heart.csv')

# Prepare the data
X = np.array(data.drop(['target'], axis=1))
y = np.array(data['target'])

# Normalize the data
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

# Convert target to binary
y_binary = (y > 0).astype(int)

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, stratify=y_binary, random_state=42, test_size=0.2)

# Load categorical model
model = load_model('../models/categorical_model.h5')

# Generate predictions for the categorical model
categorical_pred = np.argmax(model.predict(X_test), axis=1)
print('Results for Categorical Model:')
print(accuracy_score(y_test, categorical_pred))
print(classification_report(y_test, categorical_pred))

# Load binary model
binary_model = load_model('../models/binary_model.h5')

# Generate predictions for the binary model
binary_pred = np.round(binary_model.predict(X_test)).astype(int)
print('Results for Binary Model:')
print(accuracy_score(y_test, binary_pred))
print(classification_report(y_test, binary_pred))
