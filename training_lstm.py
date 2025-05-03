import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle

# --- Load data ---
X = np.load("X_padded.npy")  # Input tokenized
y = np.load("y_labels.npy")  # Output labels

# --- Load tokenizer and label encoder (optional, just for later use) ---
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# --- Split data into training and testing ---
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Build LSTM model ---
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=100, input_length=X.shape[1]))  # Embedding layer
model.add(LSTM(128, return_sequences=False))  # LSTM layer with 128 units
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# --- Compile model ---
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# --- Train model ---
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# --- Save the trained model ---
model.save("sentiment_analysis_model.h5")

print("[INFO] Model training selesai. Model disimpan sebagai 'sentiment_analysis_model.h5'.")

# --- Evaluasi model ---
loss, accuracy = model.evaluate(X_test, y_test)
print(f"[INFO] Akurasi model: {accuracy * 100:.2f}%")
