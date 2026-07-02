import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import glob

# ==========================================================
# 1. PROSES TRAINING OTOMATIS (LANGSUNG DI MEMORI)
# ==========================================================
@st.cache_resource
def inisialisasi_dan_latih_model():
    # Cari file CSV apa saja di folder utama
    csv_files = glob.glob("*.csv")
    if not csv_files:
        st.error("Waduh, tidak ada file .csv dataset sama sekali di repositori GitHub kamu!")
        return None

    target_csv = csv_files[0]
    
    # Baca dataset dengan toleransi delimiter
    try:
        df = pd.read_csv(target_csv, delimiter=';')
        if len(df.columns) <= 1:
            df = pd.read_csv(target_csv, delimiter=',')
    except Exception:
        df = pd.read_csv(target_csv)

    # Bersihkan nama kolom dari spasi liar
    df.columns = df.columns.str.strip()

    # Petakan nama kolom (fitur & target) secara fleksibel
    kolom_fitur = [col for col in df.columns if col.lower() in ['curah_hujan_mm', 'suhu_c', 'kelembapan_%', 'penyinaran_jam']]
    kolom_target = [col for col in df.columns if col.lower() in ['hasil_panen_kg']]

    if len(kolom_fitur) < 4 or not kolom_target:
        st.error("Struktur kolom dataset CSV tidak sesuai! Pastikan ada kolom Curah Hujan, Suhu, Kelembapan, Penyinaran, dan Hasil Panen.")
        return None

    X = df[kolom_fitur]
    y = df[kolom_target[0]]

    # Latih model Random Forest langsung di memori server tanpa simpan file .pkl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    
    return model_rf

# Panggil fungsi training
model = inisialisasi_dan_latih_model()

# ==========================================================
# 2. TAMPILAN APLIKASI UTAMA (STREAMLIT UI)
# ==========================================================
st.set_page_config(page_title="Prediksi Panen Cabe Baros", layout="centered")
st.title("🌾 Prediksi Hasil Panen Cabe Rawit")
st.subheader("Kecamatan Baros, Kota Sukabumi")
st.write("Masukkan indikator cuaca di bawah untuk memprediksi hasil panen (Kg).")

st.markdown("---")

if model is not None:
    # Input Cuaca oleh Pengguna
    curah_hujan = st.number_input("Curah Hujan (mm)", min_value=0.0, max_value=1000.0, value=150.0)
    suhu = st.number_input("Suhu Rata-rata (°C)", min_value=15.0, max_value=40.0, value=27.0)
    kelembapan = st.number_input("Kelembapan (%)", min_value=0.0, max_value=100.0, value=85.0)
    penyinaran = st.number_input("Lama Penyinaran Matahari (Jam)", min_value=0.0, max_value=24.0, value=5.5)

    # Tombol Eksekusi Prediksi
    if st.button("Hitung Estimasi Hasil Panen"):
        input_data = np.array([[curah_hujan, suhu, kelembapan, penyinaran]])
        prediksi = model.predict(input_data)[0]
        
        st.success("### Hasil Prediksi:")
        st.metric(label="Estimasi Hasil Panen", value=f"{prediksi:.2f} Kg")
        st.info("Prediksi dihitung menggunakan algoritma Random Forest Regressor langsung di server.")
else:
    st.warning("Aplikasi gagal memuat model. Periksa kembali file CSV Anda di GitHub.")
