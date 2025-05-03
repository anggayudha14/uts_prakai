import os
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data hasil preprocessing
file_path = os.path.join(os.getcwd(), "dataset_tiktok_comments_cleaned.csv")
df = pd.read_csv(file_path)

# Ganti NaN jadi string kosong
df = df[df['clean_text'].notnull()]
df['clean_text'] = df['clean_text'].fillna("").astype(str)
df = df[df['clean_text'].str.strip() != ""]


# --- 1. Tokenisasi ---
max_vocab = 5000  # banyak kata unik
tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
tokenizer.fit_on_texts(df['clean_text'])

sequences = tokenizer.texts_to_sequences(df['clean_text'])

# --- 2. Padding ---
max_len = 100  # panjang maksimal komentar
padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# --- 3. Encode label ke angka ---
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['label'])

# --- 4. Simpan hasil ---
import numpy as np

np.save("X_padded.npy", padded)
np.save("y_labels.npy", labels)

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("[INFO] Tokenisasi dan padding selesai.")
import joblib

# Simpan tokenizer ke file
joblib.dump(tokenizer, 'tokenizer.joblib')
print("[INFO] Tokenizer disimpan sebagai 'tokenizer.joblib'")

print(f"[INFO] Jumlah data: {len(padded)}")
print(f"[INFO] Panjang input: {padded.shape[1]}")
print(f"[INFO] Contoh sequence:\n{padded[0]}")
print(f"[INFO] Label pertama: {labels[0]}")
