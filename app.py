"""TMA Forecast Evaluator — evaluasi prediksi tinggi muka air (30 pos pemantauan).

Jalankan dengan:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from core import (
    PUBLIC_RATIO,
    SPLIT_METHODS,
    calculate_metrics,
    calculate_metrics_per_station,
    create_public_private_split,
    detect_columns,
    load_csv,
    merge_ground_truth_submission,
    top_errors,
    validate_submission,
)
from plots import (
    plot_actual_vs_prediction,
    plot_residual_distribution,
    plot_station_metrics,
    plot_time_series,
    plot_top_errors,
)

st.set_page_config(page_title="TMA Forecast Evaluator", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1250px; }
      div[data-testid="stMetric"] {
          background: rgba(130,130,130,0.06);
          border: 1px solid rgba(130,130,130,0.20);
          border-radius: 10px; padding: 12px 16px;
      }
      div[data-testid="stMetricLabel"] p { font-size: 0.78rem; opacity: 0.75; }
      div[data-testid="stMetricValue"] { font-size: 1.5rem; font-variant-numeric: tabular-nums; }
      .stTabs [data-baseweb="tab"] { padding: 8px 16px; }
      .locked { border: 1px dashed rgba(130,130,130,0.5); border-radius: 10px;
                padding: 22px; text-align: center; color: #6b6b68; }
    </style>
    """,
    unsafe_allow_html=True,
)

SAMPLE_GT = Path(__file__).parent / "sample_data" / "ground_truth.csv"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def baca(data: bytes):
    """Baca CSV (cache berdasarkan isi file)."""
    return load_csv(data)


@st.cache_data(show_spinner=False)
def gabung(gt: pd.DataFrame, sub: pd.DataFrame, a: str, b: str, c: str, d: str):
    """Merge ground truth & submission (cache)."""
    return merge_ground_truth_submission(gt, sub, a, b, c, d)


def kartu_metrik(m: dict[str, float], prefix: str = "") -> None:
    """Metric cards: RMSE, MAE, jumlah data, mean & std."""
    a, b, c = st.columns(3)
    a.metric(f"{prefix}RMSE", f"{m['rmse']:.4f} m", help="Penalti lebih besar untuk error besar. Makin kecil makin baik.")
    b.metric(f"{prefix}MAE", f"{m['mae']:.4f} m", help="Rata-rata absolute error. Makin kecil makin baik.")
    c.metric("Data dievaluasi", f"{m['n']:,}")

    d, e, f, g = st.columns(4)
    d.metric("Rata-rata aktual", f"{m['mean_aktual']:.3f} m")
    e.metric("Rata-rata prediksi", f"{m['mean_prediksi']:.3f} m")
    f.metric("Std dev aktual", f"{m['std_aktual']:.3f} m")
    g.metric("Std dev prediksi", f"{m['std_prediksi']:.3f} m")


def tabel_pos(per_pos: pd.DataFrame):
    """Tabel metrik per pos dengan highlight error tertinggi/terendah."""
    hi, lo = per_pos["rmse"].max(), per_pos["rmse"].min()

    def warnai(row):
        if row["rmse"] == hi:
            return ["background-color: rgba(227,73,72,0.14)"] * len(row)
        if row["rmse"] == lo:
            return ["background-color: rgba(0,131,0,0.12)"] * len(row)
        return [""] * len(row)

    return per_pos.style.apply(warnai, axis=1).format(
        {"rmse": "{:.4f}", "mae": "{:.4f}", "mean_aktual": "{:.3f}",
         "mean_prediksi": "{:.3f}", "n_obs": "{:,}"}
    )


def analisis(data: pd.DataFrame, ada_waktu: bool, key: str) -> None:
    """Render tab analisis (Overview / Time Series / Per Pos / Error) untuk satu dataset."""
    per_pos = calculate_metrics_per_station(data)
    t1, t2, t3, t4 = st.tabs(["Overview", "Time Series", "Per Pos", "Error Analysis"])

    with t1:
        a, b = st.columns(2)
        a.plotly_chart(plot_actual_vs_prediction(data), width="stretch", key=f"{key}_sc")
        b.plotly_chart(plot_residual_distribution(data), width="stretch", key=f"{key}_rd")

    with t2:
        if not ada_waktu:
            st.warning("Kolom waktu tidak valid — time series tidak dapat digambar.")
        else:
            pos = sorted(data["pos"].astype(str).unique())
            pilih = st.multiselect("Pos pemantauan (maks. 4 panel)", pos,
                                   default=pos[:2], key=f"{key}_pos")
            st.plotly_chart(plot_time_series(data, pilih), width="stretch", key=f"{key}_ts")

    with t3:
        st.caption("🟥 error tertinggi · 🟩 error terendah")
        st.dataframe(tabel_pos(per_pos), width="stretch", hide_index=True)
        st.download_button("⬇️ Download metrik per pos (CSV)",
                           per_pos.to_csv(index=False).encode(),
                           "metrik_per_pos.csv", "text/csv", key=f"{key}_dl")
        a, b = st.columns(2)
        a.plotly_chart(plot_station_metrics(per_pos, "rmse"), width="stretch", key=f"{key}_br")
        b.plotly_chart(plot_station_metrics(per_pos, "mae"), width="stretch", key=f"{key}_bm")

    with t4:
        st.caption("Residual = aktual − prediksi")
        sepuluh = top_errors(data, 10)
        st.plotly_chart(plot_top_errors(sepuluh), width="stretch", key=f"{key}_te")
        st.dataframe(sepuluh, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Header & sidebar
# ---------------------------------------------------------------------------

st.title("📈 TMA Forecast Evaluator")
st.caption(
    "Evaluasi prediksi time series tinggi muka air pada 30 pos pemantauan. "
    "Unggah submission, aplikasi menghitung RMSE & MAE keseluruhan dan per pos."
)

with st.sidebar:
    st.header("⚙️ Konfigurasi")
    mode = st.radio("Mode evaluasi", ["Full Analysis", "Kaggle-Style"])
    split_method = "chrono_global"
    tampil_private = "Sembunyikan"
    if mode == "Full Analysis":
        st.caption("Memakai **seluruh** ground truth untuk evaluasi.")
    else:
        st.caption(
            f"Split {PUBLIC_RATIO:.0%} **public** / {1 - PUBLIC_RATIO:.0%} **private**. "
            "Pilih metode pembagian di bawah."
        )
        split_method = st.radio(
            "Metode split",
            list(SPLIT_METHODS),
            format_func=lambda k: SPLIT_METHODS[k],
            help=(
                "**Global** membagi seluruh data; **tiap pos** membagi terpisah per pos "
                "sehingga setiap pos punya komposisi 30/70. **Random** memakai seed 42 "
                "(reproducible); **kronologis** memakai waktu paling awal."
            ),
        )
        tampil_private = st.radio("Private score", ["Sembunyikan", "Tampilkan"])

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

st.subheader("1 · Upload file")
k1, k2 = st.columns(2)
with k1:
    f_gt = st.file_uploader("Ground Truth CSV (opsional)", type=["csv"])
with k2:
    f_sub = st.file_uploader("Submission CSV", type=["csv"])

# Ground truth: pakai contoh bawaan bila tidak diunggah.
if f_gt is not None:
    gt_df, err = baca(f_gt.getvalue())
    nama_gt = f_gt.name
elif SAMPLE_GT.exists():
    gt_df, err = baca(SAMPLE_GT.read_bytes())
    nama_gt = f"{SAMPLE_GT.name} (contoh bawaan)"
    st.info(f"Ground truth tidak diunggah — memakai contoh bawaan `sample_data/{SAMPLE_GT.name}`.")
else:
    gt_df, err, nama_gt = None, "Contoh bawaan tidak ditemukan. Jalankan `python make_sample_data.py`.", ""

if err:
    st.error(f"Ground truth gagal dibaca. {err}")
    st.stop()

if f_sub is None:
    st.info("Unggah **Submission CSV** untuk memulai evaluasi.")
    with st.expander("📋 Format file yang diharapkan"):
        st.code("id,tma_mdpl\n2025-09-19 06:00:00 - Arjowinangun - Pacitan,1.38\n"
                "2025-09-19 12:00:00 - Arjowinangun - Pacitan,1.02", language="text")
        st.caption("Kolom `id` dan kolom nilai dideteksi otomatis — tidak perlu konfigurasi.")
    st.stop()

sub_df, err = baca(f_sub.getvalue())
if err:
    st.error(f"Submission gagal dibaca. {err}")
    st.stop()

id_gt, col_aktual = detect_columns(gt_df)
id_sub, col_pred = detect_columns(sub_df)

a, b = st.columns(2)
a.caption(f"**Ground truth** · `{nama_gt}` — {len(gt_df):,} baris × {len(gt_df.columns)} kolom")
a.dataframe(gt_df.head(5), width="stretch", hide_index=True)
b.caption(f"**Submission** · `{f_sub.name}` — {len(sub_df):,} baris × {len(sub_df.columns)} kolom")
b.dataframe(sub_df.head(5), width="stretch", hide_index=True)
st.caption(f"Kolom terdeteksi — identifier: `{id_gt}` / `{id_sub}` · "
           f"aktual: `{col_aktual}` · prediksi: `{col_pred}`")

# ---------------------------------------------------------------------------
# Validasi
# ---------------------------------------------------------------------------

st.subheader("2 · Validasi")
lap = validate_submission(gt_df, sub_df, id_gt, id_sub, col_pred)

for pesan in lap.errors:
    st.error(f"❌ {pesan}")
for pesan in lap.warnings:
    st.warning(f"⚠️ {pesan}")

if not lap.ok:
    st.error("**Evaluasi dihentikan.** Perbaiki submission lalu unggah ulang.")
    st.stop()

st.success(f"✅ {lap.n_cocok:,} identifier cocok dan siap dievaluasi.")

data = gabung(gt_df, sub_df, id_gt, id_sub, col_aktual, col_pred)
if data.empty:
    st.error("Hasil merge kosong — tidak ada baris dengan identifier cocok dan nilai numerik.")
    st.stop()

ada_waktu = bool(data["waktu"].notna().any())
if not ada_waktu and mode == "Kaggle-Style" and split_method.startswith("chrono"):
    st.error(
        "Metode split kronologis butuh kolom waktu yang valid, tetapi waktu tidak dapat "
        "diparse. Pilih metode **Random** di sidebar."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Evaluasi
# ---------------------------------------------------------------------------

st.subheader("3 · Hasil evaluasi")

if mode == "Full Analysis":
    kartu_metrik(calculate_metrics(data))
    st.divider()
    analisis(data, ada_waktu, "full")

else:
    d = create_public_private_split(data, split_method, PUBLIC_RATIO, seed=42)
    pub = d[d["split"] == "public"]
    pri = d[d["split"] == "private"]

    if pub.empty or pri.empty:
        st.error("Data terlalu sedikit untuk dibagi public/private.")
        st.stop()

    keterangan = {
        "chrono_global": (
            f"public = {len(pub):,} baris paling awal (s.d. "
            f"{pub['waktu'].max():%d %b %Y %H:%M}) · private = {len(pri):,} baris sesudahnya. "
            "Satu batas waktu untuk seluruh data."
        ),
        "chrono_pos": (
            f"public = {len(pub):,} baris, private = {len(pri):,} baris. "
            "Untuk **tiap pos**, 30% waktu paling awal menjadi public."
        ),
        "random_global": (
            f"public = {len(pub):,} baris, private = {len(pri):,} baris. "
            "Sampel acak 30% dari seluruh data dengan **seed 42** (reproducible)."
        ),
        "random_pos": (
            f"public = {len(pub):,} baris ({pub['pos'].nunique()} pos), private = "
            f"{len(pri):,} baris. Sampel acak 30% **tiap pos** dengan seed 42."
        ),
    }
    st.info(f"**{SPLIT_METHODS[split_method]}** · {keterangan[split_method]}")

    st.markdown("#### 🔵 Public score")
    kartu_metrik(calculate_metrics(pub), prefix="Public ")

    st.markdown("#### 🔒 Private score")
    if tampil_private == "Tampilkan":
        m_pub = calculate_metrics(pub)
        m_pri = calculate_metrics(pri)
        kartu_metrik(m_pri, prefix="Private ")
        selisih = m_pri["rmse"] - m_pub["rmse"]
        if abs(selisih) > 0.15 * max(m_pub["rmse"], 1e-9):
            st.warning(f"⚠️ Private RMSE berbeda {selisih:+.4f} m dari public — indikasi shake-up.")
        else:
            st.success(f"✅ Public & private RMSE konsisten (selisih {selisih:+.4f} m).")
    else:
        st.markdown(
            f'<div class="locked">🔒 <b>Private score disembunyikan.</b><br>'
            f"{len(pri):,} baris ({len(pri) / len(d):.0%} ground truth) dipakai sebagai "
            "private evaluation.</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("#### Analisis data public")
    analisis(pub, ada_waktu, "pub")

    if tampil_private == "Tampilkan":
        st.divider()
        st.markdown("#### Analisis data private")
        analisis(pri, ada_waktu, "pri")

st.divider()
st.caption(
    "**RMSE** memberi penalti lebih besar terhadap error besar · **MAE** = rata-rata absolute "
    "error · keduanya makin kecil makin baik. Public memakai 30% ground truth dan private 70% "
    "— performa public yang bagus tidak selalu menjamin performa private yang bagus."
)
