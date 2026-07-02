import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import os
import glob

# TRICK AMAN: Mencari file .csv apa saja yang ada di folder utama
csv_files = glob.glob("*.csv")

if not csv_files:
    raise FileNotFoundError("Waduh, tidak ada file .csv sama sekali di repositori GitHub kamu!")

# Ambil file CSV pertama yang ditemukan
target_csv = csv_files[0]
print(f"Membaca dataset: {target_csv}")

# Membaca dataset dengan toleransi pembatas (delimiter) koma atau titik koma
try:
    df = pd.read_csv(target_csv, delimiter=';')
    if len(df.columns) <= 1:
        df = pd.read_csv(target_csv, delimiter=',')
except Exception as e:
    df = pd.read_csv(target_csv)

# Membersihkan nama kolom dari spasi yang tidak sengaja terketik
df.columns = df.columns.str.strip()

# Mengambil kolom fitur & target (menggunakan pemetaan fleksibel huruf besar/kecil)
kolom_fitur = [col for col in df.columns if col.lower() in ['curah_hujan_mm', 'suhu_c', 'kelembapan_%', 'penyinaran_jam']]
kolom_target = [col for col in df.columns if col.lower() in ['hasil_panen_kg']][0]

X = df[kolom_fitur]
y = df[kolom_target]

# Latih model Random Forest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Simpan hasil latihan ke file pkl
with open('model_rf.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model berhasil dibuat otomatis dan disimpan!")
