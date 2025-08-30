# -------------------- Import Libraries -------------------- #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------- Load Dataset -------------------- #
df = pd.read_csv('/Users/nupurshivani/Downloads/Deep_Learning_Project1/Churn_Modelling.csv')
print("Original Shape:", df.shape)
print(df.head())

# -------------------- Drop Unnecessary Columns -------------------- #
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

# -------------------- Encoding Categorical Variables -------------------- #
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
print("After Encoding:\n", df.head())

# -------------------- Feature-Target Split -------------------- #
X = df.drop(columns=['Exited'])
y = df['Exited'].values

# -------------------- Train-Test Split -------------------- #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# -------------------- Feature Scaling -------------------- #
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------- Build ANN Model -------------------- #
model = Sequential()
model.add(Dense(units=11, activation='relu', input_dim=11))
model.add(Dense(units=11, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.summary()

# -------------------- Compile Model -------------------- #
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -------------------- Train Model -------------------- #
history = model.fit(X_train_scaled, y_train, batch_size=50, epochs=100, validation_split=0.2, verbose=1)

# -------------------- Plot Accuracy Graph -------------------- #
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.grid(True)
plt.show()

# -------------------- Predictions -------------------- #
y_pred = model.predict(X_test_scaled)
y_pred = (y_pred > 0.5).astype(int)

# -------------------- Evaluation -------------------- #
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
