# TIDAL KNN SBY - Streamlit Dashboard

## 1. Struktur folder

```text
TIDAL_KNN_SBY/
├── app.py
├── utils.py
├── requirements.txt
├── data/
│   └── Data_Pasut_NASA_Gabungan_WIB_Jul-Sep_2026.csv
├── models/
│   ├── knn_model.pkl
│   └── scaler_X.pkl
└── outputs/
    ├── evaluation_metrics.csv
    ├── actual_vs_prediction.csv
    ├── forecast_24_hours.csv
    ├── forecast_summary.csv
    ├── data_preprocessed.csv
    └── k_selection_results.csv
```

## 2. Instalasi

Buka terminal pada folder `TIDAL_KNN_SBY`, lalu jalankan:

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Linux/macOS:
```bash
source venv/bin/activate
```

Install library:
```bash
pip install -r requirements.txt
```

Jalankan aplikasi:
```bash
streamlit run app.py
```

## 3. File dari Google Colab yang perlu disalin

- Dataset gabungan: `Data_Pasut_NASA_Gabungan_WIB_Jul-Sep_2026.csv` ke folder `data/`.
- Evaluasi: `evaluation_metrics.csv` ke folder `outputs/`.
- Aktual vs prediksi: `actual_vs_prediction.csv` ke folder `outputs/`.
- Forecast: `forecast_24_hours.csv` ke folder `outputs/`.
- Data preprocessing: `data_preprocessed.csv` ke folder `outputs/`.
- Hasil pemilihan K: jika tersedia, simpan sebagai `k_selection_results.csv` ke folder `outputs/`.
- Model dan scaler dapat disalin ke folder `models/` untuk pengembangan prediksi langsung selanjutnya.

## 4. Catatan data dan satuan

Aplikasi mengikuti dataset penelitian SRBY Juli–September 2026. Elevasi ditampilkan dalam cm sesuai nilai dataset/laporan. Waktu ditampilkan sebagai WIB. Ringkasan metrik default mengikuti ringkasan hasil akhir laporan: K=5, MAE=10.430733 cm, RMSE=13.474547 cm, MAPE=12.627917%, R²=0.911773. Jika `evaluation_metrics.csv` tersedia, nilainya akan dibaca dari file tersebut.

## 5. Catatan kompatibilitas

Aplikasi dashboard ini membaca hasil prediksi dan evaluasi yang sudah diekspor dari Google Colab. Dashboard tidak melatih ulang model setiap kali dibuka. Ini menjaga hasil yang ditampilkan konsisten dengan eksperimen penelitian.


## Dataset yang sudah disertakan

Folder `data/` telah diisi dengan tiga file yang diberikan:
- `Data_Pasut_NASA_Gabungan_WIB_Jul-Sep_2026.csv` — 2.174 baris, 8 kolom; sumber utama dashboard.
- `Data_NASA_UTC_Jul-Sep_2026.csv` — 2.208 baris, 4 kolom.
- `Data_Pasut_UTC_DATE_TIME_HARI_ELEVATION_Jul-Sep_2026.csv` — 2.173 baris, 4 kolom.

Parsing tanggal pada dashboard disesuaikan untuk membaca format `DD/MM/YYYY` pada dataset gabungan dengan benar. File model KNN, scaler, hasil evaluasi, data aktual-vs-prediksi, dan forecast belum termasuk dalam tiga upload dataset ini; salin file hasil Google Colab tersebut ke folder `models/` dan `outputs/` bila ingin menampilkan grafik evaluasi dan forecast dari hasil eksperimen.


## Model KNN yang sudah disertakan

Model utama yang dipasang pada `models/knn_model.pkl` berasal dari upload `knn_model(1).pkl`. Metadata model menunjukkan:
- `n_neighbors = 5`
- `n_features_in_ = 60`
- matriks data latih internal berukuran `(1690, 60)`
- metrik jarak `euclidean`

Scaler utama `models/scaler_X.pkl` memiliki 60 fitur, sehingga sesuai dengan model 60-window tersebut. Ini juga sesuai dengan keterangan pipeline forecast pada laporan: K terbaik 5, window size 60, dan scaler X `scaler_X.pkl`.

File `knn_model_final.pkl`, `knn_scaler.pkl`, dan `knn_config.pkl` dipertahankan sebagai artefak alternatif/konfigurasi. Metadata `knn_model_final.pkl` menunjukkan K=2 dan 240 fitur, sedangkan `knn_scaler.pkl` memiliki 4 fitur; karena tidak cocok dengan pipeline final K=5/window 60 pada laporan, keduanya tidak dipilih sebagai model dashboard utama.

Versi scikit-learn dipatok ke 1.6.1 karena metadata artefak menunjukkan model/scaler disimpan dengan versi tersebut. Gunakan Python dan dependency yang kompatibel saat menjalankan aplikasi.


## Output penelitian yang sudah disertakan

Folder `outputs/` sudah diisi dari `outputs.zip`, termasuk:
- `evaluation_metrics.csv` dan `final_model_summary.csv` — ringkasan evaluasi model K=5 / 60 fitur.
- `actual_vs_prediction.csv` — hasil aktual dan prediksi untuk validasi/testing.
- `forecast_24_hours.csv` dan `forecast_summary.csv` — hasil forecasting 24 langkah.
- `k_selection_results.csv` dan `best_k_summary.csv` — hasil pemilihan K.
- `data_preprocessed.csv`, `preprocessing_summary.csv`, `windowing_summary.csv`, dan artefak analisis lainnya.
- Gambar hasil seperti `actual_vs_prediction_120.png` dan `forecast_combined_24_hours.png`.

Perhatian: ZIP juga memuat `knn_evaluation_metrics.csv` dan `knn_actual_vs_prediction_testing.csv` yang menunjukkan eksperimen berbeda (K=2, 4 fitur). File tersebut tetap disimpan sebagai arsip, tetapi tidak digunakan sebagai sumber metrik dashboard utama. Dashboard memprioritaskan `evaluation_metrics.csv` (K=5, 60 fitur), sesuai model dan hasil forecast utama dalam laporan.
