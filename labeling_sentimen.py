import os
import pandas as pd

dataset_name = "dataset_tiktok-comments-scraper_2025-05-01_08-04-41-083.csv"
file_path = os.path.join(os.getcwd(), dataset_name)

print("Current directory:", os.getcwd())
print("File path:", file_path)

if not os.path.isfile(file_path):
    print(f"[ERROR] File '{dataset_name}' tidak ditemukan di direktori saat ini: {os.getcwd()}")
    print("Pastikan file berada di folder yang sama dengan skrip ini, atau gunakan path lengkap.")
else:
    try:
        # Gunakan encoding yang sesuai
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        print("Kolom yang tersedia di dataset:")

        print(df.columns.tolist())

        print("[INFO] Dataset berhasil dimuat.")
        
        # Labeling sederhana
        df['label'] = df['text'].apply(lambda x: 'positive' if 'baik' in str(x).lower() else 'negative')

        df.to_csv("dataset_tiktok_comments_labelled.csv", index=False)
        print("[INFO] Hasil labeling disimpan ke 'dataset_tiktok_comments_labelled.csv'")
        print(df.head())
    except Exception as e:
        print("[ERROR] Terjadi kesalahan:", e)
