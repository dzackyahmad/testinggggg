"""Visualisasi Plotly untuk evaluasi TMA.

Palet sudah divalidasi colorblind-safe: biru = aktual, hijau = prediksi.
Identitas seri tidak pernah bergantung warna saja — legend selalu ada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

AKTUAL = "#2a78d6"
PREDIKSI = "#008300"
AKSEN = "#eb6834"
ABU = "#52514e"
GRID = "rgba(130,130,130,0.22)"

_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", size=13),
    margin=dict(l=56, r=24, t=52, b=48),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
    hoverlabel=dict(font_size=12, namelength=-1),
)
_AXIS = dict(
    showgrid=True,
    gridcolor=GRID,
    zeroline=False,
    linecolor=GRID,
    title_font=dict(color=ABU, size=12),
    tickfont=dict(color=ABU, size=11),
)


def _rapikan(fig: go.Figure, height: int = 400, **layout) -> go.Figure:
    """Terapkan layout dasar yang konsisten."""
    fig.update_layout(**_LAYOUT, height=height, **layout)
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(**_AXIS)
    return fig


def _kosong(pesan: str) -> go.Figure:
    """Placeholder saat tidak ada data."""
    fig = go.Figure()
    fig.add_annotation(text=pesan, showarrow=False, font=dict(color=ABU, size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=240, template="plotly_white")
    return fig


def plot_time_series(df: pd.DataFrame, stations: list[str]) -> go.Figure:
    """Aktual vs prediksi terhadap waktu, satu panel per pos (maks. 4).

    Small multiples dipakai agar warna selalu berarti "aktual vs prediksi"
    dan tidak pernah di-cycle antarpos.
    """
    if df.empty or not stations:
        return _kosong("Pilih minimal satu pos.")

    stations = stations[:4]
    n = len(stations)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, subplot_titles=stations,
                        vertical_spacing=0.12 / max(n - 1, 1) if n > 1 else 0.0)

    for i, pos in enumerate(stations, start=1):
        d = df[df["pos"] == pos].sort_values("waktu")
        if d.empty:
            continue
        info = np.stack([d["pos"].astype(str), d["abs_error"]], axis=-1)
        tip = (
            "<b>%{customdata[0]}</b><br>%{x|%Y-%m-%d %H:%M}<br>"
            "%{fullData.name}: %{y:.3f} m<br>Abs error: %{customdata[1]:.3f} m<extra></extra>"
        )
        for nama, kol, warna, garis in (
            ("Aktual", "aktual", AKTUAL, "solid"),
            ("Prediksi", "prediksi", PREDIKSI, "dot"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=d["waktu"], y=d[kol], name=nama, legendgroup=nama,
                    showlegend=(i == 1), mode="lines",
                    line=dict(color=warna, width=2, dash=garis),
                    customdata=info, hovertemplate=tip,
                ),
                row=i, col=1,
            )

    _rapikan(fig, height=max(280, 200 * n), hovermode="x unified")
    for a in fig.layout.annotations:
        a.update(font=dict(size=12, color=ABU), x=0, xanchor="left")
    fig.update_yaxes(title_text="TMA (m)")
    fig.update_xaxes(title_text="Waktu", row=n, col=1,
                     rangeslider=dict(visible=True, thickness=0.05))
    return fig


def plot_station_metrics(per_pos: pd.DataFrame, metric: str = "rmse") -> go.Figure:
    """Bar chart horizontal metrik per pos."""
    if per_pos.empty:
        return _kosong("Belum ada metrik per pos.")

    d = per_pos.sort_values(metric)
    fig = go.Figure(
        go.Bar(
            x=d[metric], y=d["nama_pos"], orientation="h",
            marker=dict(color=AKTUAL, line=dict(color="rgba(252,252,251,1)", width=2)),
            customdata=np.stack([d["n_obs"], d["mae"], d["rmse"]], axis=-1),
            hovertemplate=("<b>%{y}</b><br>RMSE: %{customdata[2]:.4f} m<br>"
                           "MAE: %{customdata[1]:.4f} m<br>Obs: %{customdata[0]:,}<extra></extra>"),
            text=d[metric].map("{:.3f}".format), textposition="outside", cliponaxis=False,
        )
    )
    _rapikan(fig, height=max(320, 24 * len(d) + 110), showlegend=False, bargap=0.35,
             title=dict(text=f"{metric.upper()} per pos", font=dict(size=15)))
    fig.update_xaxes(title_text=f"{metric.upper()} (m)")
    fig.update_yaxes(title_text=None, showgrid=False)
    return fig


def plot_actual_vs_prediction(df: pd.DataFrame, max_points: int = 20_000) -> go.Figure:
    """Scatter aktual vs prediksi dengan garis referensi ideal y = x."""
    if df.empty:
        return _kosong("Tidak ada data.")

    d = df.sample(max_points, random_state=42) if len(df) > max_points else df
    lo = float(min(d["aktual"].min(), d["prediksi"].min()))
    hi = float(max(d["aktual"].max(), d["prediksi"].max()))
    pad = (hi - lo) * 0.04 or 0.1

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo - pad, hi + pad], y=[lo - pad, hi + pad], mode="lines",
                             name="Prediksi sempurna (y = x)",
                             line=dict(color=ABU, width=2, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=d["aktual"], y=d["prediksi"], mode="markers", name="Observasi",
        marker=dict(size=8, color=AKTUAL, opacity=0.5,
                    line=dict(width=1, color="rgba(252,252,251,0.9)")),
        customdata=np.stack([d["pos"].astype(str), d["abs_error"]], axis=-1),
        hovertemplate=("<b>%{customdata[0]}</b><br>Aktual: %{x:.3f} m<br>"
                       "Prediksi: %{y:.3f} m<br>Abs error: %{customdata[1]:.3f} m<extra></extra>"),
    ))
    judul = f"Aktual vs prediksi ({len(d):,} titik"
    judul += f", disampel dari {len(df):,})" if len(d) < len(df) else ")"
    _rapikan(fig, height=440, title=dict(text=judul, font=dict(size=15)))
    fig.update_xaxes(title_text="Aktual (m)", range=[lo - pad, hi + pad])
    fig.update_yaxes(title_text="Prediksi (m)", range=[lo - pad, hi + pad],
                     scaleanchor="x", scaleratio=1)
    return fig


def plot_residual_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram residual (aktual - prediksi)."""
    if df.empty:
        return _kosong("Tidak ada data residual.")

    fig = go.Figure(go.Histogram(
        x=df["residual"], nbinsx=60, name="Residual",
        marker=dict(color=AKTUAL, line=dict(color="rgba(252,252,251,1)", width=1)),
        hovertemplate="Residual: %{x:.3f} m<br>Jumlah: %{y:,}<extra></extra>",
    ))
    rata = float(df["residual"].mean())
    fig.add_vline(x=0, line=dict(color="rgba(130,130,130,0.55)", width=2, dash="dash"))
    fig.add_vline(x=rata, line=dict(color=AKSEN, width=2),
                  annotation_text=f"Bias {rata:+.3f} m", annotation_position="top right",
                  annotation_font=dict(size=11, color=AKSEN))
    _rapikan(fig, height=380, showlegend=False, bargap=0.02,
             title=dict(text="Distribusi residual (aktual - prediksi)", font=dict(size=15)))
    fig.update_xaxes(title_text="Residual (m)")
    fig.update_yaxes(title_text="Jumlah observasi")
    return fig


def plot_top_errors(top: pd.DataFrame) -> go.Figure:
    """Bar chart absolute error terbesar."""
    if top.empty:
        return _kosong("Tidak ada data error.")

    d = top.copy()
    if "waktu" in d.columns and d["waktu"].notna().any():
        label = d["pos"].astype(str) + " · " + pd.to_datetime(d["waktu"]).dt.strftime("%d %b %y %H:%M")
    else:
        label = d["id"].astype(str)
    d = d.assign(label=label).sort_values("abs_error")

    fig = go.Figure(go.Bar(
        x=d["abs_error"], y=d["label"], orientation="h",
        marker=dict(color=AKSEN, line=dict(color="rgba(252,252,251,1)", width=2)),
        customdata=np.stack([d["aktual"], d["prediksi"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>Abs error: %{x:.3f} m<br>Aktual: %{customdata[0]:.3f} m"
                       "<br>Prediksi: %{customdata[1]:.3f} m<extra></extra>"),
        text=d["abs_error"].map("{:.2f}".format), textposition="outside", cliponaxis=False,
    ))
    _rapikan(fig, height=max(320, 30 * len(d) + 100), showlegend=False, bargap=0.35,
             title=dict(text=f"{len(d)} absolute error terbesar", font=dict(size=15)))
    fig.update_xaxes(title_text="Absolute error (m)")
    fig.update_yaxes(title_text=None, showgrid=False)
    return fig
