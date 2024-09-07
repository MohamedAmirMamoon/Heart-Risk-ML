import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import regularizers
import matplotlib.pyplot as plt

# Load processed data
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

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, stratify=y_binary, random_state=42, test_size=0.2)

# Build the binary classification model
def create_binary_model():
    model = Sequential()
    model.add(Dense(16, input_dim=13, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

# Train the binary model
binary_model = create_binary_model()
history = binary_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=10)

# Plot accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Binary Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Binary Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

# Save the binary model
binary_model.save('../models/binary_model.h5')
