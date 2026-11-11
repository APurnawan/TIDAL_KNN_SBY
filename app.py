import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta

from utils import (
    load_project_data, load_metrics, load_actual_prediction,
    load_forecast, load_k_search, load_preprocessed,
    normalize_datetime, get_column, build_tide_lunar_figure,
    build_actual_prediction_figure, build_forecast_figure,
    build_heatmap_figure, build_k_selection_figure, format_number
)

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Tidal Prediction Dashboard | KNN",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS - MENDEKATI DESAIN REFERENSI
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: #f7faff;
}
.block-container {
    padding-top: 1.15rem;
    padding-bottom: 1.2rem;
    max-width: 1600px;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f3f8ff 0%, #ffffff 100%);
    border-right: 1px solid #dce8f5;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.7rem;
}
.sidebar-brand {
    background: linear-gradient(120deg, #07336f, #09285a);
    color: white;
    padding: 15px 13px;
    border-radius: 8px;
    font-weight: 800;
    font-size: 15px;
    letter-spacing: .3px;
    margin-bottom: 15px;
}
.sidebar-card {
    border: 1px solid #dce7f4;
    background: rgba(255,255,255,.82);
    border-radius: 9px;
    padding: 11px 12px;
    margin: 12px 0;
    font-size: 12px;
}
.sidebar-card-title {
    color: #0756b5;
    font-size: 11px;
    font-weight: 800;
    margin-bottom: 9px;
    letter-spacing: .3px;
}
.sidebar-row {
    display:flex;
    justify-content:space-between;
    gap:8px;
    margin: 7px 0;
    color:#344967;
}
.main-title {
    color:#082e67;
    font-size:30px;
    line-height:1.15;
    font-weight:800;
    margin-bottom:5px;
}
.sub-title {
    color:#61738e;
    font-size:12px;
    margin-bottom:13px;
}
.status-pill {
    display:inline-block;
    background:#dcfce7;
    color:#168044;
    border:1px solid #b7efcb;
    border-radius:5px;
    font-size:10px;
    font-weight:800;
    padding:3px 8px;
    margin-right:7px;
}
.section-title {
    color:#082f68;
    font-weight:800;
    font-size:14px;
    border-left:4px solid #1673e6;
    padding-left:9px;
    margin: 10px 0 9px 0;
    text-transform:uppercase;
}
div[data-testid="stMetric"] {
    background:white;
    border:1px solid #e0eaf5;
    border-radius:9px;
    padding:10px 12px;
    box-shadow:0 1px 3px rgba(20,60,110,.035);
}
div[data-testid="stMetricLabel"] p {
    color:#526985;
    font-size:11px;
    font-weight:600;
}
div[data-testid="stMetricValue"] {
    color:#092f68;
    font-size:22px;
    font-weight:800;
}
div[data-testid="stMetricDelta"] {
    font-size:10px;
}
div[data-testid="stTabs"] button {
    font-size:12px;
    font-weight:600;
}
div[data-testid="stDataFrame"] {
    border:1px solid #e0eaf5;
    border-radius:8px;
}
.info-note {
    color:#64748b;
    font-size:11px;
    margin-top:3px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🌊 &nbsp; KONTROL<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown("**Periode Historis**")
    period = st.selectbox(
        "Pilih periode",
        ["7 Hari", "1 Bulan", "3 Bulan", "Seluruh Data"],
        index=2,
        label_visibility="collapsed"
    )
    st.markdown("""
    <div class="sidebar-card">
      <div class="sidebar-card-title">INFORMASI MODEL</div>
      <div class="sidebar-row"><span>Model</span><b>KNN Regression</b></div>
      <div class="sidebar-row"><span>Window Size</span><b>60 data</b></div>
      <div class="sidebar-row"><span>Training</span><b>80%</b></div>
      <div class="sidebar-row"><span>Testing</span><b>20%</b></div>
      <div class="sidebar-row"><span>Forecast</span><b>24 jam</b></div>
    </div>
    <div class="sidebar-card">
      <div class="sidebar-card-title">INFORMASI TUGAS AKHIR</div>
      <div class="sidebar-row"><span>Nama</span><b>Agus Purnawan</b></div>
      <div class="sidebar-row"><span>Universitas</span><b>Widya Kartika</b></div>
      <div class="sidebar-row"><span>Lokasi</span><b>Stasiun SRBY</b></div>
    </div>
    """, unsafe_allow_html=True)
    model_file = Path("models/knn_model.pkl")
    scaler_file = Path("models/scaler_X.pkl")
    if model_file.exists() and scaler_file.exists():
        st.success("Model KNN dan scaler tersedia")
    else:
        st.warning("File model/scaler belum lengkap")
    st.caption("Tidal Prediction Dashboard · KNN")

# ============================================================
# LOAD DATA
# ============================================================
data, data_source = load_project_data()
metrics = load_metrics()
actual_pred = load_actual_prediction()
forecast = load_forecast()
k_results = load_k_search()
preprocessed = load_preprocessed()

if data.empty:
    st.markdown('<div class="main-title">TIDAL PREDICTION DASHBOARD</div>', unsafe_allow_html=True)
    st.error("Dataset belum ditemukan. Letakkan file dataset gabungan pada folder `data/`.")
    st.code("data/Data_Pasut_NASA_Gabungan_WIB_Jul-Sep_2026.csv")
    st.info("Folder outputs/ dan models/ dapat diisi dengan file hasil Google Colab. Lihat petunjuk pada README.")
    st.stop()

data = normalize_datetime(data)
time_col = get_column(data, ["DATETIME_WIB", "DATETIME", "NASA_TIME_WIB", "DATE_TIME"])
elev_col = get_column(data, ["ELEVATION", "TINGGI_MUKA_AIR", "WATER_LEVEL"])
nasa_col = get_column(data, ["Penerangan_Persen", "PENERANGAN_PERSEN", "MOON_ILLUMINATION"])
if time_col:
    data = data.dropna(subset=[time_col]).sort_values(time_col).copy()
if elev_col:
    data[elev_col] = pd.to_numeric(data[elev_col], errors="coerce")
if nasa_col:
    data[nasa_col] = pd.to_numeric(data[nasa_col], errors="coerce")

# Filter periode historis
if time_col and not data.empty:
    latest_time = data[time_col].max()
    if period == "7 Hari":
        data_view = data[data[time_col] >= latest_time - pd.Timedelta(days=7)].copy()
    elif period == "1 Bulan":
        data_view = data[data[time_col] >= latest_time - pd.DateOffset(months=1)].copy()
    elif period == "3 Bulan":
        data_view = data[data[time_col] >= latest_time - pd.DateOffset(months=3)].copy()
    else:
        data_view = data.copy()
else:
    latest_time = None
    data_view = data.copy()

# Header
last_label = latest_time.strftime("%d %b %Y %H:%M") if latest_time is not None and pd.notna(latest_time) else "-"
st.markdown('<div class="main-title">TIDAL PREDICTION DASHBOARD</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-title"><span class="status-pill">MODEL READY</span>'
    f'Stasiun SRBY – PT PAL Indonesia, Surabaya &nbsp; | &nbsp; '
    f'Data terakhir: {last_label} WIB</div>',
    unsafe_allow_html=True
)

# ============================================================
# TOP KPI
# ============================================================
def metric_value(key, fallback="-", digits=2):
    value = metrics.get(key, fallback)
    try:
        return f"{float(value):,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(value)

if elev_col and data_view[elev_col].notna().any():
    current_elev = data_view[elev_col].dropna().iloc[-1]
    max_elev = data_view[elev_col].max()
    min_elev = data_view[elev_col].min()
    mean_elev = data_view[elev_col].mean()
else:
    current_elev = max_elev = min_elev = mean_elev = np.nan

k_best = metrics.get("best_k", metrics.get("Best K", 5))
try:
    k_best_text = str(int(float(k_best)))
except Exception:
    k_best_text = "5"

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("MUKA AIR SAAT INI", f"{current_elev:.1f} cm" if pd.notna(current_elev) else "—")
k2.metric("PREDIKSI BERIKUTNYA", f"{forecast['FORECAST_ELEVATION'].iloc[0]:.1f} cm" if not forecast.empty and "FORECAST_ELEVATION" in forecast else "—")
k3.metric("STATUS PASUT", "Menuju Pasang" if pd.notna(current_elev) else "—")
k4.metric("BEST K", k_best_text)
k5.metric("MODEL ANALITIK", "KNN")

# ============================================================
# NAVIGATION TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Analisis Historis Pasut",
    "Forecast & Evaluasi Model",
    "Analisis Pola Pasut",
    "Data Mentah (Historical Data)"
])

# ------------------------------------------------------------
# TAB 1 - HISTORICAL
# ------------------------------------------------------------
with tab1:
    st.markdown('<div class="section-title">Ringkasan Kondisi Muka Air</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Elevasi Tertinggi", f"{max_elev:.1f} cm" if pd.notna(max_elev) else "—")
    c2.metric("Elevasi Maksimum", f"{max_elev:.1f} cm" if pd.notna(max_elev) else "—")
    c3.metric("Elevasi Minimum", f"{min_elev:.1f} cm" if pd.notna(min_elev) else "—")
    c4.metric("Rata-rata", f"{mean_elev:.1f} cm" if pd.notna(mean_elev) else "—")
    c5.metric("Rentang", f"{max_elev-min_elev:.1f} cm" if pd.notna(max_elev) and pd.notna(min_elev) else "—")
    c6.metric("Std. Deviasi", f"{data_view[elev_col].std():.1f} cm" if elev_col and data_view[elev_col].notna().sum() > 1 else "—")

    st.markdown('<div class="section-title">Statistik Deskriptif Data Pasang Surut</div>', unsafe_allow_html=True)
    if elev_col:
        desc = data_view[elev_col].describe()
        stat_df = pd.DataFrame({
            "METRIK PARAMETER": ["Elevasi Tertinggi", "Elevasi Terendah", "Rata-rata Elevasi", "Standar Deviasi", "Median", "Jumlah Data"],
            "NILAI": [
                f"{desc.get('max', np.nan):.2f} cm",
                f"{desc.get('min', np.nan):.2f} cm",
                f"{desc.get('mean', np.nan):.2f} cm",
                f"{desc.get('std', np.nan):.2f} cm",
                f"{desc.get('50%', np.nan):.2f} cm",
                f"{data_view[elev_col].count():,}".replace(",", ".")
            ]
        })
        st.dataframe(stat_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Kolom elevasi tidak ditemukan pada dataset.")

    st.markdown('<div class="section-title">Grafik Pergerakan Muka Air Historis</div>', unsafe_allow_html=True)
    if elev_col and time_col:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data_view[time_col], y=data_view[elev_col],
            mode="lines", name="Elevasi Pasut",
            line=dict(color="#1673e6", width=2)
        ))
        fig.update_layout(
            height=390, margin=dict(l=10,r=10,t=20,b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            hovermode="x unified", xaxis_title="Waktu (WIB)",
            yaxis_title="Elevasi (cm)"
        )
        fig.update_xaxes(showgrid=True, gridcolor="#e8eef6")
        fig.update_yaxes(showgrid=True, gridcolor="#e8eef6")
        st.plotly_chart(fig, use_container_width=True)
    if nasa_col and time_col:
        st.markdown('<div class="section-title">Perbandingan Elevasi Pasut dan Penerangan Bulan NASA</div>', unsafe_allow_html=True)
        st.plotly_chart(build_tide_lunar_figure(data_view, time_col, elev_col, nasa_col), use_container_width=True, key="tide_lunar_historical")

# ------------------------------------------------------------
# TAB 2 - FORECAST & EVALUATION
# ------------------------------------------------------------
with tab2:
    st.markdown('<div class="section-title">Informasi Arsitektur Model</div>', unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Model Utama", "KNN Regression")
    a2.metric("Nilai K Terbaik", k_best_text)
    a3.metric("Window Size", "60 data")
    a4.metric("Horizon Forecast", "24 jam")

    st.markdown('<div class="section-title">Nilai Prediksi Horizon ke Depan</div>', unsafe_allow_html=True)
    if not forecast.empty and "FORECAST_ELEVATION" in forecast:
        fc = forecast["FORECAST_ELEVATION"].astype(float)
        p1, p2, p3 = st.columns(3)
        p1.metric("Prediksi +6 Jam", f"{fc.iloc[min(5,len(fc)-1)]:.1f} cm")
        p2.metric("Prediksi +12 Jam", f"{fc.iloc[min(11,len(fc)-1)]:.1f} cm")
        p3.metric("Prediksi +24 Jam", f"{fc.iloc[-1]:.1f} cm")
    else:
        st.info("File forecast_24_hours.csv belum tersedia. Letakkan file hasil forecast dari Google Colab pada folder outputs/.")

    st.markdown('<div class="section-title">Grafik Forecast 24 Jam ke Depan</div>', unsafe_allow_html=True)
    if not forecast.empty:
        st.plotly_chart(build_forecast_figure(data, forecast, time_col, elev_col), use_container_width=True)
    else:
        st.info("Grafik forecast akan tampil setelah file forecast tersedia.")

    st.markdown('<div class="section-title">Metrik Evaluasi Performa Model</div>', unsafe_allow_html=True)
    e1, e2, e3, e4, e5 = st.columns(5)
    # Nilai default diambil dari ringkasan hasil akhir pada laporan; file metrics akan mengutamakan nilai aktual.
    e1.metric("MAE", metric_value("mae", 10.430733, 4))
    e2.metric("RMSE", metric_value("rmse", 13.474547, 4))
    e3.metric("MAPE", f"{float(metrics.get('mape', 12.627917)):.2f}%")
    e4.metric("R² Score", metric_value("r2", 0.911773, 4))
    e5.metric("Best K", k_best_text)
    st.caption("Nilai awal mengikuti ringkasan hasil akhir laporan TA. Jika evaluation_metrics.csv tersedia, aplikasi menggunakan nilai dari file tersebut.")

    st.markdown('<div class="section-title">Visualisasi Validasi: Aktual vs Prediksi</div>', unsafe_allow_html=True)
    if not actual_pred.empty:
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(build_actual_prediction_figure(actual_pred), use_container_width=True)
        with right:
            st.dataframe(actual_pred.head(300), use_container_width=True, height=330)
    else:
        st.info("File actual_vs_prediction.csv belum tersedia.")

    st.markdown('<div class="section-title">Pemilihan Nilai K</div>', unsafe_allow_html=True)
    if not k_results.empty:
        left, right = st.columns([3, 2])
        with left:
            st.plotly_chart(build_k_selection_figure(k_results), use_container_width=True)
        with right:
            st.dataframe(k_results, use_container_width=True, height=300)
    else:
        st.info("File hasil pemilihan K belum tersedia. Aplikasi tetap menampilkan K terbaik dari hasil laporan.")

# ------------------------------------------------------------
# TAB 3 - TIDAL PATTERN
# ------------------------------------------------------------
with tab3:
    st.markdown('<div class="section-title">Analisis Pola Pasang Surut</div>', unsafe_allow_html=True)
    if elev_col:
        t1, t2, t3, t4 = st.columns(4)
        idx_max = data_view[elev_col].idxmax()
        idx_min = data_view[elev_col].idxmin()
        t1.metric("Elevasi Tertinggi", f"{data_view.loc[idx_max,elev_col]:.1f} cm")
        t2.metric("Waktu Elevasi Tertinggi", str(data_view.loc[idx_max,time_col]) if time_col else "—")
        t3.metric("Elevasi Terendah", f"{data_view.loc[idx_min,elev_col]:.1f} cm")
        t4.metric("Waktu Elevasi Terendah", str(data_view.loc[idx_min,time_col]) if time_col else "—")

    st.markdown('<div class="section-title">Pola Muka Air Berdasarkan Jam (WIB)</div>', unsafe_allow_html=True)
    if elev_col and time_col:
        hourly = data_view.copy()
        hourly["JAM"] = hourly[time_col].dt.hour
        hourly_mean = hourly.groupby("JAM", as_index=False)[elev_col].mean()
        fig_hour = go.Figure()
        fig_hour.add_trace(go.Scatter(
            x=hourly_mean["JAM"], y=hourly_mean[elev_col],
            mode="lines+markers", name="Rata-rata Elevasi",
            line=dict(color="#1673e6", width=2.5),
            marker=dict(size=6)
        ))
        fig_hour.update_layout(height=330, margin=dict(l=10,r=10,t=20,b=10),
            paper_bgcolor="white", plot_bgcolor="white",
            xaxis_title="Jam (WIB)", yaxis_title="Rata-rata Elevasi (cm)")
        fig_hour.update_xaxes(tickmode="linear", dtick=2, range=[0,23], gridcolor="#e8eef6")
        fig_hour.update_yaxes(gridcolor="#e8eef6")
        st.plotly_chart(fig_hour, use_container_width=True)

    st.markdown('<div class="section-title">Heatmap Elevasi Pasang Surut</div>', unsafe_allow_html=True)
    if elev_col and time_col:
        st.plotly_chart(build_heatmap_figure(data_view, time_col, elev_col), use_container_width=True)

    if nasa_col and time_col:
        st.markdown('<div class="section-title">Analisis Hubungan Pasut dan Penerangan Bulan</div>', unsafe_allow_html=True)
        st.plotly_chart(build_tide_lunar_figure(data_view, time_col, elev_col, nasa_col), use_container_width=True)

# ------------------------------------------------------------
# TAB 4 - RAW DATA
# ------------------------------------------------------------
with tab4:
    st.markdown('<div class="section-title">Data Mentah Pasang Surut dan NASA</div>', unsafe_allow_html=True)
    st.caption(f"Sumber data: {data_source} | Periode dataset yang ditampilkan mengikuti pilihan periode historis.")
    search = st.text_input("Cari tanggal / nilai data", placeholder="Contoh: 2026-09-30")
    raw = data_view.copy()
    if search:
        mask = raw.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        raw = raw[mask]
    st.dataframe(raw, use_container_width=True, height=520, hide_index=True)
    csv = raw.to_csv(index=False).encode("utf-8")
    st.download_button("Download data yang ditampilkan (CSV)", csv, "data_pasut_sby.csv", "text/csv")

st.divider()
st.caption(f"Tidal Prediction Dashboard | Metode K-Nearest Neighbor Regression | Data: {data_source} | Dibuat untuk keperluan penelitian Tugas Akhir.")
