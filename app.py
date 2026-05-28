import streamlit as st
import joblib
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="SMS Phishing Detection",
    page_icon="🛡️",
    layout="centered"
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model_multinomial_nb.pkl")
vectorizer = joblib.load("tfidf.pkl")

# =========================
# DOWNLOAD NLTK
# =========================
try:
    stop_words = set(stopwords.words('indonesian'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('indonesian'))

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

# =========================
# STEMMER
# =========================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# =========================
# PREPROCESSING
# =========================
def preprocessing(text):

    # 1. Lowercase
    text = text.lower()

    # 2. Remove Noise
    text = re.sub(r'http\\S+|@\\w+|#[A-Za-z0-9_]+|www\\.\\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\\s]', ' ', text)
    text = re.sub(r'\\s+', ' ', text).strip()

    # 3. Tokenize
    tokens = word_tokenize(text)

    # 4. Stopword Removal
    tokens = [w for w in tokens if w not in stop_words]

    # 5. Stemming
    text = " ".join(tokens)
    text = stemmer.stem(text)

    return text

# =========================
# UI
# =========================
st.title("🛡️ SMS Phishing Detection")
st.markdown("Deteksi SMS phishing menggunakan Machine Learning Naive Bayes")

st.divider()

input_sms = st.text_area(
    "Masukkan Isi SMS",
    height=200,
    placeholder="Contoh: Selamat anda mendapatkan hadiah..."
)

# =========================
# BUTTON
# =========================
if st.button("🔍 Analisa SMS"):

    if input_sms.strip() == "":
        st.warning("Masukkan SMS terlebih dahulu!")

    else:

        # preprocessing
        clean_text = preprocessing(input_sms)

        # vectorize
        vector = vectorizer.transform([clean_text])

        # prediction
        prediction = model.predict(vector)[0]

        # probability
        probability = model.predict_proba(vector)[0]

        genuine_prob = probability[0] * 100
        phishing_prob = probability[1] * 100

        st.divider()

        # DEBUG LABEL
        st.write("Label Prediksi:", prediction)

        # RESULT
        # GANTI sesuai label dataset kamu
        if prediction == 1:
            st.error("⚠️ SMS TERDETEKSI PHISHING")
        else:
            st.success("✅ SMS AMAN / GENUINE")

        # PERCENTAGE
        st.subheader("📊 Persentase Prediksi")

        st.write(f"Phishing : {phishing_prob:.2f}%")
        st.progress(int(phishing_prob))

        st.write(f"Genuine : {genuine_prob:.2f}%")
        st.progress(int(genuine_prob))

        # DETAIL
        st.subheader("📄 Detail Analisis")

        st.write("Hasil preprocessing:")
        st.code(clean_text)

        # STATUS BOX
        if phishing_prob >= 80:
            st.warning("Tingkat phishing sangat tinggi")
        elif phishing_prob >= 50:
            st.info("SMS cukup mencurigakan")
        else:
            st.success("SMS cenderung aman")

# =========================
# FOOTER
# =========================
st.divider()

st.caption("Machine Learning Project - Multinomial Naive Bayes")
