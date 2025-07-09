# app.py
# ==============================================================================
# TAHAP 4: PENERAPAN (DEPLOYMENT)
# ==============================================================================
# Mengimpor library yang dibutuhkan untuk aplikasi
import streamlit as st
import pandas as pd
import joblib

# --- Fungsi untuk Memuat Model dan Kolom ---
# Menggunakan cache agar model tidak perlu dimuat ulang setiap kali ada interaksi
@st.cache_resource
def load_model_assets():
    """Memuat model dan daftar nama kolom yang telah dilatih."""
    try:
        model = joblib.load('random_forest_smote_model.joblib')
        model_cols = joblib.load('model_columns.joblib')
        return model, model_cols
    except FileNotFoundError:
        return None, None

# --- Memuat Aset ---
model, model_columns = load_model_assets()

# --- Tampilan Utama Aplikasi ---
st.set_page_config(page_title="Prediksi Attrition", layout="wide")
st.title("👨‍💼 Aplikasi Prediksi Attrition Karyawan")
st.markdown("Aplikasi ini adalah implementasi dari model machine learning yang telah dilatih untuk memprediksi kemungkinan seorang karyawan akan keluar dari perusahaan.")

# --- Logika Aplikasi ---
# Jika model tidak berhasil dimuat, tampilkan pesan error
if model is None or model_columns is None:
    st.error("File model (`random_forest_smote_model.joblib`) atau kolom (`model_columns.joblib`) tidak ditemukan. Pastikan Anda telah menjalankan notebook Tahap 1-3 dan file-file tersebut berada di folder yang sama dengan aplikasi ini.")
else:
    st.header("Masukkan Data Karyawan untuk Prediksi Interaktif")
    # Membuat kolom untuk layout yang rapi
    col1, col2, col3 = st.columns(3)

    # Mengumpulkan input dari pengguna
    with col1:
        Age = st.slider('Usia', 18, 60, 35)
        MonthlyIncome = st.slider('Pendapatan Bulanan ($)', 1000, 20000, 5000)
    with col2:
        TotalWorkingYears = st.slider('Total Tahun Bekerja', 0, 40, 10)
        YearsAtCompany = st.slider('Tahun di Perusahaan Ini', 0, 40, 5)
    with col3:
        OverTime = st.selectbox('Bekerja Lembur?', ['Yes', 'No'], index=1)
        JobSatisfaction = st.select_slider('Tingkat Kepuasan Kerja', options=[1, 2, 3, 4], value=3)

    # Tombol untuk memicu proses prediksi
    if st.button('✨ Buat Prediksi', type="primary"):
        # Membuat DataFrame kosong dengan struktur kolom yang sama persis seperti saat pelatihan
        input_data = pd.DataFrame(columns=model_columns)
        input_data.loc[0] = 0 # Mengisi semua nilai awal dengan 0

        # Memasukkan nilai dari input pengguna ke kolom yang sesuai
        input_data['Age'] = Age
        input_data['MonthlyIncome'] = MonthlyIncome
        input_data['TotalWorkingYears'] = TotalWorkingYears
        input_data['YearsAtCompany'] = YearsAtCompany
        input_data['JobSatisfaction'] = JobSatisfaction
        if OverTime == 'Yes':
            input_data['OverTime_Yes'] = 1

        # Melakukan prediksi menggunakan model yang telah dimuat
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # Menampilkan hasil prediksi
        st.subheader('Hasil Prediksi:')
        if prediction[0] == 1:
            st.error(f'Karyawan ini **BERISIKO TINGGI** untuk keluar.', icon="⚠️")
            st.metric(label="Tingkat Keyakinan (Probabilitas Keluar)", value=f"{prediction_proba[0][1]*100:.2f}%")
        else:
            st.success(f'Karyawan ini kemungkinan besar akan **BERTAHAN**.', icon="✅")
            st.metric(label="Tingkat Keyakinan (Probabilitas Bertahan)", value=f"{prediction_proba[0][0]*100:.2f}%")