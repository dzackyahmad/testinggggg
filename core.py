"""Logika inti evaluasi TMA: load, validasi, merge, metrik, split kronologis.

Murni Pandas/NumPy — tidak menggambar UI, sehingga mudah diuji terpisah.
"""

from __future__ import annotations

import io
import zlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

PUBLIC_RATIO = 0.30

# Pemisah composite id: "<datetime> - <nama_pos>".
# nama_pos sendiri boleh mengandung " - " (mis. "Arjowinangun - Pacitan"),
# sehingga pemecahan dibatasi pada kemunculan pertama saja.
ID_SEPARATOR = " - "


# ---------------------------------------------------------------------------
# Load & deteksi kolom
# ---------------------------------------------------------------------------


def load_csv(data: bytes) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Baca CSV dari bytes dengan deteksi separator otomatis.

    Mengembalikan ``(dataframe, pesan_error)`` — salah satunya selalu None.
    """
    if not data:
        return None, "File kosong (0 byte)."

    terbaik: Optional[pd.DataFrame] = None
    for sep in (",", ";", "\t", "|"):
        try:
            df = pd.read_csv(io.BytesIO(data), sep=sep)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(io.BytesIO(data), sep=sep, encoding="latin-1")
            except Exception:  # noqa: BLE001
                continue
        except Exception:  # noqa: BLE001
            continue
        # Separator yang benar menghasilkan kolom terbanyak.
        if terbaik is None or len(df.columns) > len(terbaik.columns):
            terbaik = df

    if terbaik is None:
        return None, "CSV tidak dapat dibaca — format tidak dikenali."
    if len(terbaik.columns) < 2:
        return None, "Hanya 1 kolom terbaca — separator CSV tidak sesuai."
    if terbaik.empty:
        return None, "File tidak memiliki baris data."
    return terbaik, None


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Tebak kolom identifier dan kolom nilai secara otomatis (tanpa UI).

    Identifier: kolom bernama ``id`` bila ada, selain itu kolom pertama.
    Nilai: kolom ``tma_mdpl`` bila ada, selain itu kolom numerik terakhir.
    """
    kolom = list(map(str, df.columns))
    id_col = next((c for c in kolom if c.lower() == "id"), kolom[0])

    nilai = next((c for c in kolom if c.lower() == "tma_mdpl"), None)
    if nilai is None:
        numerik = [c for c in kolom if c != id_col and pd.api.types.is_numeric_dtype(df[c])]
        nilai = numerik[-1] if numerik else kolom[-1]
    return id_col, nilai


def parse_composite_id(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Pecah ``"<datetime> - <nama_pos>"`` menjadi (waktu, pos).

    Pemecahan memakai ``n=1`` agar nama pos yang mengandung ` - ` tetap utuh.
    """
    parts = series.astype(str).str.split(ID_SEPARATOR, n=1, expand=True)
    if parts.shape[1] < 2:
        parts[1] = np.nan
    return pd.to_datetime(parts[0], errors="coerce"), parts[1].astype("string")


# ---------------------------------------------------------------------------
# Validasi
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Error (blocking) dan warning (non-blocking) hasil validasi submission."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_cocok: int = 0
    n_missing: int = 0
    n_extra: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_submission(
    gt: pd.DataFrame, sub: pd.DataFrame, id_gt: str, id_sub: str, pred_col: str
) -> ValidationReport:
    """Validasi submission terhadap ground truth sebelum evaluasi."""
    rep = ValidationReport()

    gt_id = gt[id_gt].astype(str).str.strip()
    sub_id = sub[id_sub].astype(str).str.strip()

    n_dup = int(sub_id.duplicated().sum())
    if n_dup:
        contoh = sub_id[sub_id.duplicated()].unique()[:3]
        rep.errors.append(
            f"Terdapat **{n_dup:,}** identifier duplikat. Contoh: {', '.join(contoh)}"
        )

    pred = sub[pred_col]
    pred_num = pd.to_numeric(pred, errors="coerce")
    n_kosong = int(pred.isna().sum())
    n_nonnum = int((pred_num.isna() & pred.notna()).sum())
    n_inf = int(np.isinf(pred_num.to_numpy(dtype="float64", na_value=0.0)).sum())

    if n_kosong:
        rep.errors.append(f"Terdapat **{n_kosong:,}** nilai prediksi kosong.")
    if n_nonnum:
        contoh = pred[pred_num.isna() & pred.notna()].unique()[:3]
        rep.errors.append(
            f"Terdapat **{n_nonnum:,}** nilai prediksi nonnumerik. "
            f"Contoh: {', '.join(map(str, contoh))}"
        )
    if n_inf:
        rep.errors.append(f"Terdapat **{n_inf:,}** nilai prediksi tak hingga (inf).")

    set_gt, set_sub = set(gt_id), set(sub_id)
    rep.n_cocok = len(set_gt & set_sub)
    rep.n_missing = len(set_gt - set_sub)
    rep.n_extra = len(set_sub - set_gt)

    if not rep.n_cocok:
        rep.errors.append(
            "Tidak ada identifier yang cocok dengan ground truth — "
            "pastikan submission memakai kolom `id` yang sama."
        )
    elif rep.n_missing:
        rep.errors.append(
            f"**{rep.n_missing:,}** identifier ground truth tidak ada pada submission "
            f"(dari total {len(set_gt):,})."
        )
    if rep.n_extra:
        rep.warnings.append(
            f"{rep.n_extra:,} identifier tidak dikenal di ground truth — diabaikan."
        )
    return rep


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_ground_truth_submission(
    gt: pd.DataFrame,
    sub: pd.DataFrame,
    id_gt: str,
    id_sub: str,
    actual_col: str,
    pred_col: str,
) -> pd.DataFrame:
    """Gabungkan berdasarkan **identifier**, bukan urutan baris.

    Waktu dan nama pos diambil dari kolom ``datetime``/``nama_pos`` bila tersedia,
    selain itu diturunkan dari composite id. Kolom hasil: ``id``, ``waktu``,
    ``pos``, ``aktual``, ``prediksi``, ``residual``, ``abs_error``.
    """
    kiri = pd.DataFrame(
        {
            "id": gt[id_gt].astype(str).str.strip(),
            "aktual": pd.to_numeric(gt[actual_col], errors="coerce"),
        }
    )

    kolom_gt = {str(c).lower(): c for c in gt.columns}
    if "datetime" in kolom_gt and "nama_pos" in kolom_gt:
        kiri["waktu"] = pd.to_datetime(gt[kolom_gt["datetime"]], errors="coerce")
        kiri["pos"] = gt[kolom_gt["nama_pos"]].astype("string")
    else:
        kiri["waktu"], kiri["pos"] = parse_composite_id(kiri["id"])

    kanan = pd.DataFrame(
        {
            "id": sub[id_sub].astype(str).str.strip(),
            "prediksi": pd.to_numeric(sub[pred_col], errors="coerce"),
        }
    ).drop_duplicates(subset="id", keep="first")

    m = kiri.merge(kanan, on="id", how="inner").dropna(subset=["aktual", "prediksi"])
    m["pos"] = m["pos"].fillna("(tidak diketahui)")
    m["residual"] = m["aktual"] - m["prediksi"]
    m["abs_error"] = m["residual"].abs()
    return m.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metrik
# ---------------------------------------------------------------------------


def calculate_metrics(df: pd.DataFrame) -> dict[str, float]:
    """RMSE, MAE, jumlah data, serta mean & std aktual/prediksi.

    RMSE = sqrt(mean((aktual - prediksi)^2)) · MAE = mean(abs(aktual - prediksi))
    """
    if df.empty:
        return dict.fromkeys(
            ("rmse", "mae", "mean_aktual", "mean_prediksi", "std_aktual", "std_prediksi"),
            float("nan"),
        ) | {"n": 0}

    a = df["aktual"].to_numpy(dtype="float64")
    p = df["prediksi"].to_numpy(dtype="float64")
    err = a - p
    return {
        "n": int(len(df)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "mean_aktual": float(a.mean()),
        "mean_prediksi": float(p.mean()),
        "std_aktual": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "std_prediksi": float(p.std(ddof=1)) if len(p) > 1 else 0.0,
    }


def calculate_metrics_per_station(df: pd.DataFrame) -> pd.DataFrame:
    """RMSE & MAE untuk setiap pos pemantauan, diurutkan dari error terbesar."""
    if df.empty:
        return pd.DataFrame(
            columns=["nama_pos", "n_obs", "rmse", "mae", "mean_aktual", "mean_prediksi"]
        )

    g = df.groupby("pos", observed=True, dropna=False)
    out = pd.DataFrame(
        {
            "n_obs": g.size(),
            "rmse": g.apply(
                lambda x: float(np.sqrt(np.mean((x["aktual"] - x["prediksi"]) ** 2))),
                include_groups=False,
            ),
            "mae": g["abs_error"].mean(),
            "mean_aktual": g["aktual"].mean(),
            "mean_prediksi": g["prediksi"].mean(),
        }
    ).reset_index(names="nama_pos")
    return out.sort_values("rmse", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public - Private split
# ---------------------------------------------------------------------------


def _k_public(n: int, ratio: float) -> int:
    """Jumlah baris public: dibulatkan, minimal 1 & maksimal n-1 bila n >= 2."""
    if n < 2:
        return n
    return min(max(int(round(n * ratio)), 1), n - 1)


def _stable_seed(seed: int, teks: str) -> int:
    """Seed turunan deterministik lintas proses (crc32, bukan hash() bawaan)."""
    return (seed + zlib.crc32(teks.encode("utf-8"))) & 0xFFFFFFFF


# Empat metode split public-private (urutan = urutan tampil; pertama = default).
SPLIT_METHODS = {
    "random_global": "Random — seed 42 (global) · default kaggle",
    "random_pos": "Random — seed 42 tiap pos",
    "chrono_global": "Kronologis — 30% data paling awal (global)",
    "chrono_pos": "Kronologis — 30% paling awal tiap pos",
}


def create_public_private_split(
    df: pd.DataFrame,
    method: str = "random_global",
    public_ratio: float = PUBLIC_RATIO,
    seed: int = 42,
) -> pd.DataFrame:
    """Bagi data menjadi public/private dengan salah satu dari empat metode.

    - ``chrono_global`` — urutkan seluruh data per waktu, 30% baris awal = public.
    - ``chrono_pos``    — untuk tiap pos, 30% waktu paling awal = public.
    - ``random_global`` — sampel acak 30% dari seluruh data (seed tetap).
    - ``random_pos``    — sampel acak 30% per pos (seed tetap, turunan per pos).

    Semua metode menjamin public & private **disjoint**. Metode per-pos menjamin
    setiap pos (>= 2 obs) punya minimal 1 baris di masing-masing bagian.
    Menambahkan kolom ``split`` bernilai ``"public"``/``"private"``.
    """
    if df.empty:
        return df.assign(split=pd.Series(dtype="object"))

    if method == "chrono_global":
        d = df.sort_values("waktu", kind="mergesort", na_position="last").reset_index(
            drop=True
        )
        k = _k_public(len(d), public_ratio)
        d["split"] = "private"
        d.loc[: k - 1, "split"] = "public"
        return d

    if method == "random_global":
        d = df.reset_index(drop=True).copy()
        k = _k_public(len(d), public_ratio)
        pilih = np.random.default_rng(seed).choice(d.index.to_numpy(), size=k, replace=False)
        d["split"] = "private"
        d.loc[pilih, "split"] = "public"
        return d

    # --- metode per pos (chrono_pos / random_pos) ---
    d = df.reset_index(drop=True).copy()
    d["split"] = "private"
    public_idx: list = []
    for pos, grup in d.groupby("pos", observed=True, sort=True):
        n = len(grup)
        if n == 1:
            public_idx.extend(grup.index.tolist())  # pos 1 obs -> public agar terlihat
            continue
        k = _k_public(n, public_ratio)
        if method == "chrono_pos":
            urut = grup.sort_values("waktu", kind="mergesort", na_position="last")
            public_idx.extend(urut.index[:k].tolist())
        else:  # random_pos
            rng = np.random.default_rng(_stable_seed(seed, str(pos)))
            public_idx.extend(
                rng.choice(grup.index.to_numpy(), size=k, replace=False).tolist()
            )
    d.loc[public_idx, "split"] = "public"
    return d


def top_errors(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """``n`` prediksi dengan absolute error terbesar."""
    if df.empty:
        return df
    kolom = ["id", "waktu", "pos", "aktual", "prediksi", "abs_error"]
    return df.nlargest(n, "abs_error")[[c for c in kolom if c in df.columns]].reset_index(
        drop=True
    )
