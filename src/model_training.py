import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import regularizers
import matplotlib.pyplot as plt

# Load the processed data
data = pd.read_csv('../dataset/processed_heart.csv')

# Prepare the data
X = np.array(data.drop(['target'], axis=1))
y = np.array(data['target'])

# Normalize the data
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

# Convert to categorical labels
from keras.utils.np_utils import to_categorical
Y_train = to_categorical(y_train)
Y_test = to_categorical(y_test)

# Build the model
def create_model():
    model = Sequential()
    model.add(Dense(16, input_dim=13, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

# Train the model
model = create_model()
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=10)

# Plot accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()

# Save the model
model.save('../models/categorical_model.h5')
