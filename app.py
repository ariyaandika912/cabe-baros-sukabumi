import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# TRICK: Jika model belum terbentuk di server, jalankan train.py otomatis
if not os.path.exists('model_rf.pkl'):
    os.system('python train.py')

# Load model
@st.cache_resource
def load_model():
    with open('model_rf.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Tampilan Aplikasi
st.set_page_config(page_title="Prediksi Panen Cabe Baros", layout="centered")
st.title("🌾 Prediksi Hasil Panen Cabe Rawit")
st.subheader("Kecamatan Baros, Kota Sukabumi")
st.write("Masukkan indikator cuaca di bawah untuk memprediksi hasil panen (Kg).")

st.markdown("---")

# Input Cuaca
curah_hujan = st.number_input("Curah Hujan (mm)", min_value=0.0, max_value=1000.0, value=150.0)
suhu = st.number_input("Suhu Rata-rata (°C)", min_value=15.0, max_value=40.0, value=27.0)
kelembapan = st.number_input("Kelembapan (%)", min_value=0.0, max_value=100.0, value=85.0)
penyinaran = st.number_input("Lama Penyinaran Matahari (Jam)", min_value=0.0, max_value=24.0, value=5.5)

# Tombol Eksekusi
if st.button("Hitung Estimasi Hasil Panen"):
    input_data = np.array([[curah_hujan, suhu, kelembapan, penyinaran]])
    prediksi = model.predict(input_data)[0]
    
    st.success("### Hasil Prediksi:")
    st.metric(label="Estimasi Hasil Panen", value=f"{prediksi:.2f} Kg")
