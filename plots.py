"""Visualisasi TMA sebagai gambar statis (matplotlib).

Dirender di server jadi PNG lewat ``st.pyplot`` — browser hanya menerima gambar,
tanpa JavaScript, sehingga halaman ringan. Setiap figure dibuat dengan OO API
``Figure`` (bukan ``plt`` global) agar tidak membocorkan memori di server.

Palet: biru = aktual, hijau = prediksi. Label angka ditampilkan langsung pada
chart agar nilai mudah dibaca tanpa interaksi.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# Tema gelap agar serasi dengan halaman hitam & angka tetap terbaca.
BG = "#000000"          # latar chart: hitam
FG = "#e8e8e8"          # teks utama: terang
AKTUAL = "#3987e5"      # biru terang (aktual)
PREDIKSI = "#33c02f"    # hijau terang (prediksi) — kontras di atas hitam
AKSEN = "#ff7a45"       # oranye (aksen error/bias)
ABU = "#a8a8a8"         # teks sekunder
GRID = "#333333"        # grid halus

# Font sedikit lebih besar agar angka terbaca jelas pada gambar statis.
_FONT = {"title": 13, "label": 11, "tick": 10, "value": 9.5}


def _bersih(ax) -> None:
    """Rapikan sumbu: latar hitam, grid tipis, hilangkan spine berlebih."""
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for sisi in ("top", "right"):
        ax.spines[sisi].set_visible(False)
    for sisi in ("left", "bottom"):
        ax.spines[sisi].set_color(GRID)
    ax.tick_params(colors=ABU, labelsize=_FONT["tick"], length=0)


def _fig(figsize) -> Figure:
    """Buat Figure berlatar hitam."""
    fig = Figure(figsize=figsize, dpi=120)
    fig.patch.set_facecolor(BG)
    return fig


def _kosong(pesan: str) -> Figure:
    """Placeholder saat tidak ada data."""
    fig = _fig((7, 2.6))
    ax = fig.subplots()
    ax.set_facecolor(BG)
    ax.text(0.5, 0.5, pesan, ha="center", va="center", color=ABU, fontsize=12)
    ax.axis("off")
    return fig


def _label_batang_h(ax, nilai, ypos, fmt="{:.3f}", dx=None) -> None:
    """Tulis nilai di ujung kanan tiap batang horizontal."""
    lebar = ax.get_xlim()[1]
    if dx is None:
        dx = lebar * 0.01
    for v, y in zip(nilai, ypos):
        ax.text(v + dx, y, fmt.format(v), va="center", ha="left",
                fontsize=_FONT["value"], color=ABU)


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------


def plot_time_series(df: pd.DataFrame, stations: list[str]) -> Figure:
    """Aktual vs prediksi terhadap waktu, satu panel per pos (maks. 4)."""
    if df.empty or not stations:
        return _kosong("Pilih minimal satu pos.")

    stations = stations[:4]
    n = len(stations)
    fig = _fig((11, 2.5 * n + 0.4))
    axes = fig.subplots(n, 1, sharex=True, squeeze=False)[:, 0]

    for ax, pos in zip(axes, stations):
        d = df[df["pos"] == pos].sort_values("waktu")
        if not d.empty:
            ax.plot(d["waktu"], d["aktual"], color=AKTUAL, lw=1.8, label="Aktual")
            ax.plot(d["waktu"], d["prediksi"], color=PREDIKSI, lw=1.6,
                    ls="--", label="Prediksi")
        ax.set_title(str(pos), fontsize=_FONT["label"], color=ABU, loc="left", pad=4)
        ax.set_ylabel("TMA (m)", fontsize=_FONT["label"], color=ABU)
        _bersih(ax)

    leg = axes[0].legend(loc="upper right", fontsize=_FONT["tick"], ncol=2,
                         facecolor="#141414", edgecolor=GRID, labelcolor=FG)
    leg.get_frame().set_alpha(0.95)
    axes[-1].set_xlabel("Waktu", fontsize=_FONT["label"], color=ABU)
    fig.autofmt_xdate(rotation=25, ha="right")
    fig.tight_layout(pad=1.1)
    return fig


# ---------------------------------------------------------------------------
# Metrik per pos
# ---------------------------------------------------------------------------


def plot_station_metrics(per_pos: pd.DataFrame, metric: str = "rmse") -> Figure:
    """Bar chart horizontal metrik per pos, dengan label nilai di tiap batang."""
    if per_pos.empty:
        return _kosong("Belum ada metrik per pos.")

    d = per_pos.sort_values(metric)
    y = np.arange(len(d))
    fig = _fig((7.2, 0.32 * len(d) + 1.3))
    ax = fig.subplots()

    ax.barh(y, d[metric], color=AKTUAL, height=0.68, edgecolor=BG, linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(d["nama_pos"], fontsize=_FONT["tick"])
    ax.set_xlim(0, float(d[metric].max()) * 1.16)
    ax.set_xlabel(f"{metric.upper()} (m)", fontsize=_FONT["label"], color=ABU)
    ax.set_title(f"{metric.upper()} per pos", fontsize=_FONT["title"],
                 color=FG, loc="left", pad=8, fontweight="bold")
    _bersih(ax)
    ax.grid(axis="y", visible=False)
    _label_batang_h(ax, d[metric].to_numpy(), y, fmt="{:.4f}")
    fig.tight_layout(pad=1.0)
    return fig


# ---------------------------------------------------------------------------
# Aktual vs prediksi
# ---------------------------------------------------------------------------


def plot_actual_vs_prediction(df: pd.DataFrame, max_points: int = 20_000) -> Figure:
    """Scatter aktual vs prediksi + garis ideal y = x + kotak ringkasan RMSE/MAE."""
    if df.empty:
        return _kosong("Tidak ada data.")

    d = df.sample(max_points, random_state=42) if len(df) > max_points else df
    a = d["aktual"].to_numpy(); p = d["prediksi"].to_numpy()
    lo = float(min(a.min(), p.min())); hi = float(max(a.max(), p.max()))
    pad = (hi - lo) * 0.04 or 0.1
    rmse = float(np.sqrt(np.mean((a - p) ** 2))); mae = float(np.mean(np.abs(a - p)))

    fig = _fig((6.4, 5.4))
    ax = fig.subplots()
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=ABU, lw=1.6,
            ls="--", label="Prediksi sempurna (y = x)")
    ax.scatter(a, p, s=14, color=AKTUAL, alpha=0.4, edgecolor="none", label="Observasi")

    ax.set_title("Aktual vs prediksi", fontsize=_FONT["title"], color=FG,
                 loc="left", pad=8, fontweight="bold")
    ax.set_xlabel("Aktual (m)", fontsize=_FONT["label"], color=ABU)
    ax.set_ylabel("Prediksi (m)", fontsize=_FONT["label"], color=ABU)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    _bersih(ax)
    leg = ax.legend(loc="lower right", fontsize=_FONT["tick"],
                    facecolor="#141414", edgecolor=GRID, labelcolor=FG)
    leg.get_frame().set_alpha(0.95)
    catatan = f"RMSE = {rmse:.4f} m\nMAE  = {mae:.4f} m\n{len(d):,} titik"
    if len(d) < len(df):
        catatan += f" (dari {len(df):,})"
    ax.text(0.03, 0.97, catatan, transform=ax.transAxes, va="top", ha="left",
            fontsize=_FONT["label"], color=FG,
            bbox=dict(boxstyle="round,pad=0.4", fc="#141414", ec=GRID, alpha=0.95))
    fig.tight_layout(pad=1.0)
    return fig


# ---------------------------------------------------------------------------
# Residual
# ---------------------------------------------------------------------------


def plot_residual_distribution(df: pd.DataFrame) -> Figure:
    """Histogram residual (aktual - prediksi) dengan garis nol & garis bias."""
    if df.empty:
        return _kosong("Tidak ada data residual.")

    r = df["residual"].to_numpy()
    rata = float(r.mean()); std = float(r.std())
    fig = _fig((6.4, 4.6))
    ax = fig.subplots()
    ax.hist(r, bins=60, color=AKTUAL, edgecolor=BG, linewidth=0.4)
    ax.axvline(0, color=ABU, lw=1.6, ls="--")
    ax.axvline(rata, color=AKSEN, lw=2.0)

    ymax = ax.get_ylim()[1]
    ax.text(rata, ymax * 0.96, f" bias {rata:+.3f} m", color=AKSEN,
            fontsize=_FONT["label"], va="top", ha="left", fontweight="bold")
    ax.set_title("Distribusi residual (aktual - prediksi)", fontsize=_FONT["title"],
                 color=FG, loc="left", pad=8, fontweight="bold")
    ax.set_xlabel("Residual (m)", fontsize=_FONT["label"], color=ABU)
    ax.set_ylabel("Jumlah observasi", fontsize=_FONT["label"], color=ABU)
    _bersih(ax)
    ax.text(0.97, 0.97, f"mean {rata:+.3f} m\nstd  {std:.3f} m",
            transform=ax.transAxes, va="top", ha="right", fontsize=_FONT["tick"],
            color=ABU,
            bbox=dict(boxstyle="round,pad=0.35", fc="#141414", ec=GRID, alpha=0.95))
    fig.tight_layout(pad=1.0)
    return fig


# ---------------------------------------------------------------------------
# Top error
# ---------------------------------------------------------------------------


def plot_top_errors(top: pd.DataFrame) -> Figure:
    """Bar chart absolute error terbesar, dengan label nilai di tiap batang."""
    if top.empty:
        return _kosong("Tidak ada data error.")

    d = top.copy()
    if "waktu" in d.columns and d["waktu"].notna().any():
        label = (d["pos"].astype(str) + " · "
                 + pd.to_datetime(d["waktu"]).dt.strftime("%d %b %y %H:%M"))
    else:
        label = d["id"].astype(str)
    d = d.assign(label=label).sort_values("abs_error")
    y = np.arange(len(d))

    fig = _fig((8.6, 0.42 * len(d) + 1.3))
    ax = fig.subplots()
    ax.barh(y, d["abs_error"], color=AKSEN, height=0.66, edgecolor=BG, linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"], fontsize=_FONT["tick"])
    ax.set_xlim(0, float(d["abs_error"].max()) * 1.14)
    ax.set_xlabel("Absolute error (m)", fontsize=_FONT["label"], color=ABU)
    ax.set_title(f"{len(d)} absolute error terbesar", fontsize=_FONT["title"],
                 color=FG, loc="left", pad=8, fontweight="bold")
    _bersih(ax)
    ax.grid(axis="y", visible=False)
    _label_batang_h(ax, d["abs_error"].to_numpy(), y, fmt="{:.3f}")
    fig.tight_layout(pad=1.0)
    return fig
