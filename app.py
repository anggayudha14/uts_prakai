from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import os
import re
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model dan tokenizer
model = load_model('sentiment_analysis_model.h5')
tokenizer = joblib.load('tokenizer.joblib')
max_length = 100

# Stopwords Bahasa Indonesia
stop_words = set(stopwords.words('indonesian'))

# Preprocessing untuk wordcloud
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

# Prediksi sentimen
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_length)
    pred = model.predict(pad)[0][0]
    return "Positif" if pred >= 0.5 else "Negatif"

@app.route('/', methods=['GET', 'POST'])
def index():
    data = None
    hasil_text = None
    wordcloud = False

    if request.method == 'POST':
        if 'comment' in request.form:
            comment = request.form['comment']
            hasil_text = predict_sentiment(comment)

        elif 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if file.filename.endswith('.csv'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                try:
                    df = pd.read_csv(filepath)
                except Exception as e:
                    return f"Gagal membaca CSV: {e}", 400

                df = df.rename(columns=lambda x: x.strip().lower())

                if 'komentar' in df.columns:
                    comment_col = 'komentar'
                elif 'comment' in df.columns:
                    comment_col = 'comment'
                else:
                    comment_col = df.columns[0]  # fallback

                # Tambah komentar dummy jika kosong semua
                if df[comment_col].dropna().str.strip().eq('').all():
                    df = pd.DataFrame({
                        comment_col: [
                            "Kami bersama sukatani.",
                            "Sukatani berkata benar.",
                            "Kayanya grup band sukatani gak bisa bayar nih hingga minta maaf.",
                            "Polisi adalah hama.",
                            "Di bungkam karena fakta.",
                            "Tugas jibril memberi wahyu, tugas sukatani memberi fakta.",
                            "Rip kebebasan berekspresi.",
                            "Mereka bukan klarifikasi tapi diintimidasi.",
                            "Realita yang dibungkam.",
                            "Ngapain minta maaf?"
                            "Tersinggung artinya?"
                            "Karya seni dibungkam=lawan."
                            "Bapak gue polisi gue setel didepan bapak gue hahahaha."
                            "Yang bikin susah kita mengekspresikan kritikan kita kepada pemerintah UU ITE.Padahal lagunya itu kritikan harusnya lembaga  yang dikritik ya berbenah."
                            "Intinya ya bayar polisi."
                            "Kalian mewakili rakyat."
                            "Kocak polri."
                            "Klarifikasi karena belum bayar polisi."
                            "Purbalingga."
                            "Padahal fakta."
                            "KAMIBERSAMASUKATANI."
                            "Ngapain minta maaf"
                            "Dan lagi karya dibatasi."
                            "Kalo bicara fakta memang susah di konoha."
                            "kenapa?jujur itu susah."
                            "Negara kocak."
                            "Lah kan emang kalo gak bayar gak diurus."
                            "Bayar bayar bayar."
                            "Sama lirik doang panik."
                        ]
                    })

                # Analisis sentimen
                df['sentimen'] = df[comment_col].astype(str).apply(predict_sentiment)
                df.to_csv('uploaded/hasil_analisis.csv', index=False)

                # WordCloud
                cleaned_comments = [preprocess_text(c) for c in df[comment_col].astype(str)]
                all_text = " ".join(cleaned_comments)

                custom_stopwords = set(STOPWORDS)
                custom_stopwords.update(['tiktok', 'https', 'com', 'jpg', 'dr', 'id'])

                wc = WordCloud(width=800, height=400, background_color='white',
                               stopwords=custom_stopwords).generate(all_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig('static/wordcloud.png')
                wordcloud = True

                data = df.to_dict(orient='records')

    return render_template('index.html', hasil_text=hasil_text, data=data, wordcloud=wordcloud)

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    text = request.form['input_text']
    hasil_text = predict_sentiment(text)
    return render_template('index.html', hasil_text=hasil_text)

@app.route('/download')
def download():
    return send_file('uploaded/hasil_analisis.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
