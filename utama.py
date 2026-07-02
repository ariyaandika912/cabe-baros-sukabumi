import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import glob

# ==========================================================
# 1. PROSES TRAINING OTOMATIS (6 VARIABEL)
# ==========================================================
@st.cache_resource
def inisialisasi_dan_latih_model():
    csv_files = glob.glob("*.csv")
    if not csv_files:
        st.error("Waduh, tidak ada file .csv dataset sama sekali di repositori GitHub kamu!")
        return None

    target_csv = csv_files[0]
    
    try:
        df = pd.read_csv(target_csv, delimiter=';')
        if len(df.columns) <= 1:
            df = pd.read_csv(target_csv, delimiter=',')
    except Exception:
        df = pd.read_csv(target_csv)

    df.columns = df.columns.str.strip()

    # PILIHAN FILTER NAMA KOLOM (Sesuaikan dengan nama kolom di CSV kamu)
    # Menambahkan pencarian untuk tahun dan minggu secara fleksibel
    kolom_tahun = [col for col in df.columns if col.lower() in ['tahun', 'thn', 'year']]
    kolom_minggu = [col for col in df.columns if col.lower() in ['minggu', 'minggu_ke', 'week', 'mgg']]
    kolom_ch = [col for col in df.columns if col.lower() in ['curah_hujan_mm', 'curah_hujan']]
    kolom_suhu = [col for col in df.columns if col.lower() in ['suhu_c', 'suhu']]
    kolom_kelembapan = [col for col in df.columns if col.lower() in ['kelembapan_%', 'kelembapan']]
    kolom_penyinaran = [col for col in df.columns if col.lower() in ['penyinaran_jam', 'penyinaran']]
    
    kolom_target = [col for col in df.columns if col.lower() in ['hasil_panen_kg', 'hasil_panen']]

    # Validasi apakah kolom yang dicari lengkap
    if not (kolom_tahun and kolom_minggu and kolom_ch and kolom_suhu and kolom_kelembapan and kolom_penyinaran) or not kolom_target:
        st.error("Struktur kolom dataset CSV tidak sesuai! Pastikan memiliki kolom: Tahun, Minggu, Curah Hujan, Suhu, Kelembapan, Penyinaran, dan Hasil Panen.")
        return None

    # Urutan fitur yang dimasukkan ke model latihan
    fitur_terpilih = [kolom_tahun[0], kolom_minggu[0], kolom_ch[0], kolom_suhu[0], kolom_kelembapan[0], kolom_penyinaran[0]]
    
    X = df[fitur_terpilih]
    y = df[kolom_target[0]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    
    return model_rf

model = inisialisasi_dan_latih_model()

# ==========================================================
# 2. TAMPILAN APLIKASI UTAMA (STREAMLIT UI)
# ==========================================================
st.set_page_config(page_title="Prediksi Panen Cabe Baros", layout="centered")
st.title("🌾 Prediksi Hasil Panen Cabe Rawit")
st.subheader("Kecamatan Baros, Kota Sukabumi")
st.write("Masukkan indikator di bawah untuk memprediksi hasil panen (Kg).")

st.markdown("---")

if model is not None:
    # INPUT BARU: TAHUN DAN MINGGU
    col1, col2 = st.columns(2)
    with col1:
        tahun = st.number_input("Tahun", min_value=2010, max_value=2030, value=2026)
    with col2:
        minggu = st.number_input("Minggu Ke-", min_value=1, max_value=53, value=1)

    st.markdown("---")

    # Input Cuaca Lama
    curah_hujan = st.number_input("Curah Hujan (mm)", min_value=0.0, max_value=1000.0, value=150.0)
    suhu = st.number_input("Suhu Rata-rata (°C)", min_value=15.0, max_value=40.0, value=27.0)
    kelembapan = st.number_input("Kelembapan (%)", min_value=0.0, max_value=100.0, value=85.0)
    penyinaran = st.number_input("Lama Penyinaran Matahari (Jam)", min_value=0.0, max_value=24.0, value=5.5)

    # Tombol Eksekusi Prediksi
    if st.button("Hitung Estimasi Hasil Panen"):
        # Susunan array input disesuaikan dengan fitur saat latihan (Tahun, Minggu, CH, Suhu, Kelembapan, Penyinaran)
        input_data = np.array([[tahun, minggu, curah_hujan, suhu, kelembapan, penyinaran]])
        prediksi = model.predict(input_data)[0]
        
        st.success("### Hasil Prediksi:")
        st.metric(label="Estimasi Hasil Panen", value=f"{prediksi:.2f} Kg")
        st.info("Prediksi dihitung berdasarkan variabel waktu dan indikator cuaca menggunakan Random Forest.")
else:
    st.warning("Aplikasi gagal memuat model. Periksa kembali file CSV Anda di GitHub.")
