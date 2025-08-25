#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import shap
import joblib

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Pengiriman", layout="wide")

# Load pipeline dan komponen
try:
    pipeline_model = joblib.load('pipeline_model_binary.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    preprocessor = pipeline_model.named_steps['preprocessor']
    model = pipeline_model.named_steps['classifier']
    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.TreeExplainer(model)
except FileNotFoundError:
    st.error("File model atau label encoder tidak ditemukan. Pastikan file `.pkl` sudah diunggah.")
    st.stop()

# Gunakan st.container() untuk membungkus seluruh konten utama
with st.container():
    # Judul utama
    st.title("📦 Prediksi Performa Pengiriman")
    st.markdown("Isi data pengiriman di bawah, lalu klik tombol prediksi untuk mengetahui apakah pengiriman akan **Tepat Waktu** atau **Terlambat**.")

    # Input user
    with st.expander("📝 Input Data Pengiriman", expanded=True):
        with st.form("input_form"):
            col1, col2 = st.columns(2)
            with col1:
                weather = st.selectbox('Cuaca', ['Windy', 'Sandstorms', 'Cloudy', 'Sunny', 'Fog', 'Stormy'])
                vehicle_type = st.selectbox('Kendaraan', ['scooter', 'motorcycle', 'electric_scooter'])
                city = st.selectbox('Kota', ['Urban', 'Semi-Urban', 'Metropolitan'])
                festival = st.selectbox('Festival', ['No', 'Yes'])
                traffic_density = st.selectbox('Lalu Lintas', ['Low', 'Medium', 'High', 'Jam'])
            with col2:
                age = st.number_input('Usia Pengemudi', 20, 40, 25)
                ratings = st.number_input('Rating Pengemudi', 1.0, 5.0, 4.0)
                distance = st.number_input('Jarak (km)', 0.1, 30.0, 5.0)
                multiple_deliveries = st.number_input('Multiple Deliveries', 0, 5, 0)
                vehicle_condition = st.number_input('Kondisi Kendaraan (0-2)', 0, 2, 1)

            submitted = st.form_submit_button("🔍 Prediksi Sekarang")

    # Proses prediksi
    if submitted:
        input_data = pd.DataFrame([{
            'Weather_conditions': weather,
            'Type_of_vehicle': vehicle_type,
            'City': city,
            'Festival': festival,
            'Road_traffic_density': traffic_density,
            'Delivery_person_Age': age,
            'Delivery_person_Ratings': ratings,
            'Jarak': distance,
            'multiple_deliveries': multiple_deliveries,
            'Vehicle_condition': vehicle_condition
        }])

        processed = preprocessor.transform(input_data)
        pred = model.predict(processed)
        label = label_encoder.inverse_transform(pred)[0]

        # Hasil prediksi
        st.subheader("📊 Hasil Prediksi")
        if label == "Tepat Waktu":
            st.success(f"Performa Pengiriman: **{label}** 🎉")
        else:
            st.warning(f"Performa Pengiriman: **{label}** ⚠️")

            # SHAP audit internal
            try:
                shap_vals = explainer.shap_values(processed)[0]
                shap_abs = pd.Series(shap_vals, index=feature_names).abs()
                top_feat = shap_abs.sort_values(ascending=False).index[0]

                # Interpretasi pemicu utama
                if traffic_density in ['High', 'Jam']:
                    top_clean = 'Kepadatan Lalu Lintas'
                elif weather in ['Stormy', 'Sandstorms']:
                    top_clean = 'Kondisi Cuaca'
                elif 'scaler__Jarak' in top_feat and distance < 5:
                    top_feat = shap_abs.sort_values(ascending=False).index[1] if len(shap_abs) > 1 else top_feat
                    top_clean = top_feat.split('__')[-1]
                else:
                    top_clean = top_feat.split('__')[-1]

                # Insight hanya jika prediksi = Terlambat
                st.subheader("🔥 Pemicu Utama")
                st.markdown(f"Prediksi ini paling dipengaruhi oleh: **{top_clean}**")

                st.subheader("✅ Rekomendasi Aksi")
                if 'Kepadatan Lalu Lintas' in top_clean:
                    st.markdown("- **Sarankan rute alternatif** untuk menghindari kemacetan.")
                elif 'Jarak' in top_feat:
                    st.markdown("- **Optimalkan rute** atau gunakan pengemudi yang lokasinya lebih dekat.")
                elif 'Delivery_person_Age' in top_feat:
                    st.markdown("- **Berikan pelatihan** atau panduan rute untuk pengemudi ini.")
                elif 'Delivery_person_Ratings' in top_feat:
                    st.markdown("- **Berikan insentif** atau umpan balik kepada pengemudi untuk meningkatkan performa.")
                elif 'Weather_conditions' in top_feat:
                    st.markdown("- **Beri pengemudi peringatan cuaca** dan estimasi waktu yang lebih realistis.")
                else:
                    st.markdown("- Faktor lain memengaruhi. Analisis lebih lanjut mungkin diperlukan.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat menjelaskan hasil prediksi: {e}")
