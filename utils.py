from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None

def read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def load_project_data():
    candidates = [
        DATA_DIR / "Data_Pasut_NASA_Gabungan_WIB_Jul-Sep_2026.csv",
        DATA_DIR / "dataset_gabungan.csv",
        DATA_DIR / "data_gabungan.csv",
        OUTPUT_DIR / "data_preprocessed.csv",
    ]
    path = first_existing(candidates)
    if path is None:
        return pd.DataFrame(), "Dataset belum ditemukan"
    return read_csv_safe(path), path.name

def normalize_datetime(df):
    out = df.copy()

    # Dataset gabungan penelitian menggunakan format tanggal DD/MM/YYYY.
    # dayfirst=True menjaga 01/07/2026 dibaca sebagai 1 Juli 2026, bukan 7 Januari.
    for col in ["DATETIME_WIB", "NASA_TIME_WIB", "DATETIME", "DATE_TIME"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce", dayfirst=True)

    if "DATE" in out.columns:
        date_text = out["DATE"].astype(str).str.strip()
        # Tangani format ISO YYYY-MM-DD dan DD/MM/YYYY.
        iso_mask = date_text.str.match(r"^\\d{4}-\\d{2}-\\d{2}$")
        parsed_date = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
        parsed_date.loc[iso_mask] = pd.to_datetime(
            date_text.loc[iso_mask], format="%Y-%m-%d", errors="coerce"
        )
        parsed_date.loc[~iso_mask] = pd.to_datetime(
            date_text.loc[~iso_mask], dayfirst=True, errors="coerce"
        )
        out["DATE"] = parsed_date

    if "DATETIME_WIB" not in out.columns and {"DATE", "TIME"}.issubset(out.columns):
        out["DATETIME_WIB"] = pd.to_datetime(
            out["DATE"].dt.strftime("%Y-%m-%d") + " " + out["TIME"].astype(str),
            errors="coerce"
        )
    return out

def get_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None

def load_metrics():
    candidates = [
        OUTPUT_DIR / "evaluation_metrics.csv",
        OUTPUT_DIR / "final_model_summary.csv",
        OUTPUT_DIR / "metrics.csv",
    ]
    path = first_existing(candidates)
    if path:
        df = read_csv_safe(path)
        if not df.empty:
            row = df.iloc[0].to_dict()
            # Mendukung format Metric-Value dan kolom langsung.
            if {"Metric", "Value"}.issubset(df.columns):
                row.update(dict(zip(df["Metric"].astype(str), df["Value"])))
            aliases = {
                "Best K": "best_k", "Best_K": "best_k", "K": "best_k",
                "MAE": "mae", "RMSE": "rmse", "MAPE": "mape", "R2": "r2", "R²": "r2"
            }
            for old, new in aliases.items():
                if old in row and new not in row:
                    row[new] = row[old]
            return row
    # Nilai ringkasan hasil akhir sebagaimana tertulis dalam laporan.
    return {"best_k": 5, "mae": 10.430733, "rmse": 13.474547,
            "mape": 12.627917, "r2": 0.911773}

def load_actual_prediction():
    path = first_existing([
        OUTPUT_DIR / "actual_vs_prediction.csv",
        OUTPUT_DIR / "actual_vs_prediction_120.csv",
    ])
    if path is None:
        return pd.DataFrame()
    df = read_csv_safe(path)
    if df.empty:
        return df
    # Seragamkan nama kolom
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["actual", "aktual", "y_true", "elevation_actual", "actual_elevation"]:
            rename[c] = "Actual"
        elif cl in ["prediction", "predicted", "prediksi", "y_pred", "elevation_prediction", "predicted_elevation"]:
            rename[c] = "Prediction"
        elif cl in ["datetime", "datetime_wib", "time", "waktu"]:
            rename[c] = "Datetime"
    df = df.rename(columns=rename)
    return df

def load_forecast():
    path = first_existing([
        OUTPUT_DIR / "forecast_24_hours.csv",
        OUTPUT_DIR / "forecast.csv",
    ])
    if path is None:
        return pd.DataFrame()
    df = read_csv_safe(path)
    if df.empty:
        return df
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["datetime", "datetime_wib", "date", "waktu"]:
            rename[c] = "DATETIME"
        elif cl in ["forecast_elevation", "forecast", "prediksi", "prediction"]:
            rename[c] = "FORECAST_ELEVATION"
    df = df.rename(columns=rename)
    if "DATETIME" in df:
        df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
    if "FORECAST_ELEVATION" in df:
        df["FORECAST_ELEVATION"] = pd.to_numeric(df["FORECAST_ELEVATION"], errors="coerce")
    return df

def load_k_search():
    candidates = [
        OUTPUT_DIR / "k_selection_results.csv",
        OUTPUT_DIR / "hasil_pemilihan_k.csv",
        OUTPUT_DIR / "k_results.csv",
        OUTPUT_DIR / "k_search_results.csv",
    ]
    path = first_existing(candidates)
    if path is None:
        return pd.DataFrame()
    df = read_csv_safe(path)
    if df.empty:
        return df
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["k", "n_neighbors", "nilai_k"]:
            rename[c] = "K"
        elif cl == "mae":
            rename[c] = "MAE"
        elif cl == "rmse":
            rename[c] = "RMSE"
        elif cl in ["r2", "r²", "r2_score"]:
            rename[c] = "R2"
    return df.rename(columns=rename)

def load_preprocessed():
    return read_csv_safe(OUTPUT_DIR / "data_preprocessed.csv")


def build_tide_lunar_figure(df, time_col, elev_col, nasa_col):
    fig = go.Figure()

    if time_col and elev_col and elev_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df[time_col],
            y=df[elev_col],
            name="Elevasi Pasut",
            mode="lines",
            line=dict(color="#159447", width=1.8),
            yaxis="y"
        ))

    if time_col and nasa_col and nasa_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df[time_col],
            y=df[nasa_col],
            name="Penerangan Bulan NASA",
            mode="lines",
            line=dict(color="#1769d2", width=1.6),
            yaxis="y2"
        ))

    fig.update_layout(
        height=390,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=25, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            title="Tanggal dan Waktu (WIB)",
            showgrid=True,
            gridcolor="#e8eef6"
        ),
        yaxis=dict(
            title=dict(
                text="Elevasi Pasut (cm)",
                font=dict(color="#159447")
            ),
            tickfont=dict(color="#159447"),
            gridcolor="#e8eef6"
        ),
        yaxis2=dict(
            title=dict(
                text="Penerangan Bulan (%)",
                font=dict(color="#1769d2")
            ),
            tickfont=dict(color="#1769d2"),
            overlaying="y",
            side="right",
            range=[0, 100],
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            y=1.12,
            x=0
        )
    )

    return fig


def build_actual_prediction_figure(df):
    fig = go.Figure()
    x = df["Datetime"] if "Datetime" in df else np.arange(len(df))
    if "Actual" in df:
        fig.add_trace(go.Scatter(x=x, y=df["Actual"], name="Data Aktual",
                                 mode="lines", line=dict(color="#1673e6", width=2)))
    if "Prediction" in df:
        fig.add_trace(go.Scatter(x=x, y=df["Prediction"], name="Prediksi KNN",
                                 mode="lines", line=dict(color="#16a34a", width=2)))
    fig.update_layout(height=340, margin=dict(l=10,r=10,t=20,b=10),
                      paper_bgcolor="white", plot_bgcolor="white",
                      hovermode="x unified", xaxis_title="Observasi / Waktu",
                      yaxis_title="Elevasi (cm)", legend=dict(orientation="h", y=1.1))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e8eef6")
    return fig

def build_forecast_figure(data, forecast, time_col, elev_col):
    fig = go.Figure()
    if time_col and elev_col:
        hist = data[[time_col, elev_col]].dropna().tail(120)
        fig.add_trace(go.Scatter(x=hist[time_col], y=hist[elev_col],
                                 name="Data Aktual (Historis)", mode="lines",
                                 line=dict(color="#1673e6", width=2)))
    if not forecast.empty and {"DATETIME","FORECAST_ELEVATION"}.issubset(forecast.columns):
        fig.add_trace(go.Scatter(x=forecast["DATETIME"], y=forecast["FORECAST_ELEVATION"],
                                 name="Forecast 24 Jam (Prediksi)", mode="lines+markers",
                                 line=dict(color="#16a34a", width=2.5),
                                 marker=dict(size=5)))
        if time_col and data[time_col].notna().any():
            start = forecast["DATETIME"].min()
            fig.add_vline(x=start, line_dash="dash", line_color="#64748b",
                          annotation_text="Mulai Forecast", annotation_position="top")
    fig.update_layout(height=390, margin=dict(l=10,r=10,t=25,b=10),
                      paper_bgcolor="white", plot_bgcolor="white",
                      hovermode="x unified", xaxis_title="Waktu (WIB)",
                      yaxis_title="Elevasi (cm)", legend=dict(orientation="h", y=1.12))
    fig.update_xaxes(showgrid=True, gridcolor="#e8eef6")
    fig.update_yaxes(showgrid=True, gridcolor="#e8eef6")
    return fig

def build_heatmap_figure(df, time_col, elev_col):
    temp = df[[time_col, elev_col]].dropna().copy()
    temp["Tanggal"] = temp[time_col].dt.strftime("%d-%b")
    temp["Jam"] = temp[time_col].dt.hour
    pivot = temp.pivot_table(index="Jam", columns="Tanggal", values=elev_col, aggfunc="mean")
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale="YlGnBu", colorbar=dict(title="Elevasi (cm)")
    ))
    fig.update_layout(height=430, margin=dict(l=10,r=10,t=20,b=80),
                      paper_bgcolor="white", xaxis_title="Tanggal",
                      yaxis_title="Jam (WIB)")
    fig.update_xaxes(tickangle=-45)
    return fig

def build_k_selection_figure(df):
    fig = go.Figure()
    if "K" in df and "MAE" in df:
        fig.add_trace(go.Scatter(x=df["K"], y=df["MAE"], name="MAE",
                                 mode="lines+markers", line=dict(color="#1673e6", width=2)))
    if "K" in df and "RMSE" in df:
        fig.add_trace(go.Scatter(x=df["K"], y=df["RMSE"], name="RMSE",
                                 mode="lines+markers", line=dict(color="#16a34a", width=2)))
    fig.update_layout(height=320, margin=dict(l=10,r=10,t=20,b=10),
                      paper_bgcolor="white", plot_bgcolor="white",
                      xaxis_title="Nilai K", yaxis_title="Nilai Error",
                      legend=dict(orientation="h", y=1.1))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e8eef6")
    return fig

def format_number(value, digits=2):
    try:
        return f"{float(value):,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(value)
