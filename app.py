import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Spacer, Image

# Load model dan tampilkan akurasi
model = joblib.load('model_rf.pkl')
mape_value = 12.20  # Nilai MAPE model
rmse_value = 204890.42  # Nilai RMSE model
model_accuracy = 87.8

# Konfigurasi tema dark mode pada matplotlib
plt.style.use('dark_background')

# Judul aplikasi dan informasi performa model
st.title("Prediksi Viewers Video Channel YouTube Warganet Life Official")
st.write(f"##### Akurasi Model: {model_accuracy}%")
st.write(f"\nMAPE: {mape_value}% \nRMSE: {rmse_value}")
st.markdown("""Disclaimer: Nilai prediksi belum tentu akurat, tergantung pada nilai MAPE dan RMSE.""")

# Opsi input
st.sidebar.subheader("Pilih metode input data")
input_option = st.sidebar.radio("Metode Input", ("Input Manual", "Unggah CSV"))

# Fungsi prediksi
def predict(data):
    predictions = model.predict(data)
    return predictions

# Fungsi untuk membuat file Excel dari hasil prediksi
def generate_excel(data_csv):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Tulis data ke Excel
        data_csv.to_excel(writer, index=False, sheet_name='Hasil Prediksi')
        
        # Atur worksheet dan format kolom agar sesuai dengan konten
        workbook = writer.book
        worksheet = writer.sheets['Hasil Prediksi']
        
        # Tentukan format untuk judul
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'center',
            'fg_color': '#D7E4BC',
            'border': 1
        })

        # Terapkan format untuk header
        for col_num, value in enumerate(data_csv.columns.values):
            worksheet.write(0, col_num, value, header_format)
            worksheet.set_column(col_num, col_num, 20)  # Lebar kolom default, dapat disesuaikan

        # Set kolom agar menyesuaikan konten
        for i, col in enumerate(data_csv.columns):
            max_length = max(data_csv[col].astype(str).apply(len).max(), len(col)) + 2
            worksheet.set_column(i, i, max_length)

    output.seek(0)
    return output

# Input data manual
if input_option == "Input Manual":
    # Input manual dari user
    pembagian = st.number_input("Pembagian", min_value=0)
    tidak_suka = st.number_input("Tidak suka", min_value=0)
    suka = st.number_input("Suka", min_value=0)
    subscriber_hilang = st.number_input("Subscriber yang hilang", min_value=0)
    subscriber_diperoleh = st.number_input("Subscriber yang diperoleh", min_value=0)
    waktu_tonton = st.number_input("Waktu tonton (jam)", min_value=0)
    subscriber = st.number_input("Subscriber", min_value=0)
    tayangan = st.number_input("Tayangan", min_value=0)

    input_data = pd.DataFrame({
        'Waktu tonton (jam)': [waktu_tonton],
        'Pembagian': [pembagian],
        'Tidak suka': [tidak_suka],
        'Suka': [suka],
        'Subscriber yang hilang': [subscriber_hilang],
        'Subscriber yang diperoleh': [subscriber_diperoleh],
        'Subscriber': [subscriber],
        'Tayangan': [tayangan]
    })

    if st.button("Prediksi Penayangan"):
        prediction = predict(input_data)[0]
        st.session_state.prediksi_penayangan = prediction  # Simpan prediksi ke session state

    # Menampilkan hasil prediksi jika sudah ada
    if "prediksi_penayangan" in st.session_state:
        prediction = st.session_state.prediksi_penayangan
        st.markdown(f"<h3>Hasil Prediksi Penayangan: {int(prediction):,}</h3>", unsafe_allow_html=True)

        # Input nilai aktual untuk perbandingan manual
        actual_views = st.number_input("Masukkan Nilai Aktual Penayangan untuk Evaluasi", min_value=0)
        if actual_views:
            # Evaluasi dengan RMSE
            if abs(prediction - actual_views) > rmse_value and actual_views < prediction:
                st.write("**Perlu Evaluasi**: Penayangan aktual jauh di bawah nilai prediksi, pertimbangkan untuk evaluasi konten.")
            else:
                st.write("Penayangan aktual sesuai dengan prediksi, tidak diperlukan evaluasi konten.")

            # Visualisasi perbandingan menggunakan matplotlib
            fig, ax = plt.subplots()
            ax.bar(['Prediksi', 'Aktual'], [prediction, actual_views], color=['#1f77b4', '#ff7f0e'])
            ax.set_ylabel('Penayangan')
            ax.set_title('Perbandingan Prediksi dan Aktual')
            st.pyplot(fig)

# Input dari file CSV
else:
    st.subheader("Unggah File CSV")
    st.write("Pastikan urutan kolom csv: ")
    st.markdown("""
        1. Judul video
        2. Waktu tonton (jam)
        3. Pembagian
        4. Tidak suka
        5. Suka
        6. Subscriber yang hilang
        7. Subscriber yang diperoleh
        8. Subscriber
        9. Tayangan
        10. Penayangan
    """)
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")

    if uploaded_file is not None:
        data_csv = pd.read_csv(uploaded_file)
        expected_columns = ['Judul video', 'Waktu tonton (jam)', 'Pembagian', 'Tidak suka', 'Suka', 
                            'Subscriber yang hilang', 'Subscriber yang diperoleh', 'Subscriber', 
                            'Tayangan', 'Penayangan']

        if all(column in data_csv.columns for column in expected_columns):
            predictions = predict(data_csv[expected_columns[1:-1]])
            data_csv['Prediksi Penayangan'] = predictions

            # Evaluasi menggunakan RMSE
            data_csv['Perlu Evaluasi'] = data_csv.apply(
                lambda row: 'Ya' if (abs(row['Prediksi Penayangan'] - row['Penayangan']) > rmse_value and row['Penayangan'] < row['Prediksi Penayangan']) else 'Tidak', 
                axis=1
            )

            # Format kolom 'Penayangan' dan 'Prediksi Penayangan'
            data_csv['Penayangan'] = data_csv['Penayangan'].apply(lambda x: f"{int(x):,}")
            data_csv['Prediksi Penayangan'] = data_csv['Prediksi Penayangan'].apply(lambda x: f"{int(x):,}")

            # Tampilkan dan beri opsi download hasil prediksi
            st.write("**Hasil Prediksi**")
            st.dataframe(data_csv[['Judul video', 'Penayangan', 'Prediksi Penayangan', 'Perlu Evaluasi']], height=300)

            # Visualisasi dengan matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            bar_width = 0.35
            index = np.arange(len(data_csv))

            ax.bar(index, data_csv['Penayangan'].apply(lambda x: int(x.replace(',', ''))), bar_width, label='Aktual', color='orange')
            ax.bar(index + bar_width, data_csv['Prediksi Penayangan'].apply(lambda x: int(x.replace(',', ''))), bar_width, label='Prediksi', color='blue')

            ax.set_xlabel("Indeks Data")
            ax.set_ylabel("Penayangan")
            ax.set_title("Perbandingan Penayangan Prediksi dan Aktual")
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels([f"Data {i+1}" for i in data_csv.index], rotation=45, ha='right')
            ax.legend()

            st.pyplot(fig)

            excel_file = generate_excel(data_csv)
            st.download_button(
                label="Unduh Hasil Prediksi sebagai Excel",
                data=excel_file,
                file_name="hasil_prediksi.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        else:
            st.write("**Error:** Pastikan file CSV memiliki kolom yang sesuai.")
