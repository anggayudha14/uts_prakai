import os
import pandas as pd
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Pastikan file hasil labeling sebelumnya ada
file_path = os.path.join(os.getcwd(), "dataset_tiktok_comments_labelled.csv")

if not os.path.isfile(file_path):
    print("[ERROR] File hasil labeling tidak ditemukan.")
else:
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    print("[INFO] Dataset berhasil dimuat.")

    # Inisialisasi stopwords Bahasa Indonesia
    stop_words = set(stopwords.words('indonesian'))

    def preprocess_text(text):
        if pd.isna(text):
            return ""
        text = text.lower()  # lowercase
        text = re.sub(r"http\S+|www.\S+", "", text)  # hapus URL
        text = re.sub(r"[^a-zA-Z\s]", "", text)  # hapus simbol & angka
        text = text.translate(str.maketrans("", "", string.punctuation))  # hapus tanda baca
        tokens = text.split()  # tokenisasi
        filtered = [word for word in tokens if word not in stop_words]  # hapus stopwords
        return " ".join(filtered)

    # Terapkan preprocessing
    df['clean_text'] = df['text'].apply(preprocess_text)

    # Simpan ke file baru
    df.to_csv("dataset_tiktok_comments_cleaned.csv", index=False)
    print("[INFO] Preprocessing selesai. Disimpan sebagai 'dataset_tiktok_comments_cleaned.csv'")
    print(df[['text', 'clean_text', 'label']].head())
