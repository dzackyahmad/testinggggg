"""Buat contoh ground_truth.csv dan submission.csv untuk mencoba aplikasi.

Ground truth diambil dari ``test.csv` (kolom `id`,`tma_mdpl`) bila tersedia;
bila tidak, data sintetis 30 pos dibangkitkan. Submission dibuat dengan
menambahkan noise + bias pada nilai aktual sehingga RMSE/MAE tidak nol.

Pemakaian:
    python make_sample_data.py [path/ke/test.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
OUT = Path(__file__).parent / "sample_data"


def synthetic_ground_truth(n_pos: int = 30, n_step: int = 240) -> pd.DataFrame:
    """Bangkitkan ground truth sintetis berformat composite id."""
    rng = np.random.default_rng(SEED)
    kabupaten = ["Pacitan", "Ngawi", "Madiun", "Kediri", "Blitar", "Malang"]
    pos = [f"Pos-{i:02d} - {kabupaten[i % len(kabupaten)]}" for i in range(1, n_pos + 1)]
    waktu = pd.date_range("2025-09-19 06:00:00", periods=n_step, freq="6h")

    baris = []
    for p in pos:
        dasar = rng.uniform(0.6, 3.2)
        musim = np.sin(np.linspace(0, 6 * np.pi, n_step)) * rng.uniform(0.1, 0.5)
        acak = np.cumsum(rng.normal(0, 0.03, n_step))
        tma = np.clip(dasar + musim + acak, 0.05, None)
        for t, v in zip(waktu, tma):
            baris.append(
                {"id": f"{t:%Y-%m-%d %H:%M:%S} - {p}", "tma_mdpl": round(float(v), 2)}
            )
    return pd.DataFrame(baris)


def make_submission(gt: pd.DataFrame) -> pd.DataFrame:
    """Turunkan submission realistis dari ground truth (noise + bias per pos)."""
    rng = np.random.default_rng(SEED + 1)
    pos = gt["id"].str.split(" - ", n=1).str[1]

    # Bias berbeda tiap pos agar metrik per pos bervariasi.
    bias_pos = {p: rng.normal(0, 0.08) for p in pos.unique()}
    bias = pos.map(bias_pos).to_numpy()
    noise = rng.normal(0, 0.12, len(gt))

    pred = gt["tma_mdpl"].to_numpy() + bias + noise
    # Sedikit outlier besar agar RMSE terlihat berbeda jelas dari MAE.
    idx = rng.choice(len(gt), size=max(1, len(gt) // 200), replace=False)
    pred[idx] += rng.normal(0, 1.2, len(idx))

    return pd.DataFrame(
        {"id": gt["id"], "tma_mdpl": np.round(np.clip(pred, 0.01, None), 3)}
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)

    sumber = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "test.csv"
    if sumber.exists():
        gt = pd.read_csv(sumber)[["id", "tma_mdpl"]].copy()
        print(f"Ground truth dari {sumber} — {len(gt):,} baris.")
    else:
        gt = synthetic_ground_truth()
        print(f"{sumber} tidak ditemukan — memakai data sintetis ({len(gt):,} baris).")

    sub = make_submission(gt)
    gt.to_csv(OUT / "ground_truth.csv", index=False)
    sub.to_csv(OUT / "submission.csv", index=False)

    err = gt["tma_mdpl"].to_numpy() - sub["tma_mdpl"].to_numpy()
    print(f"Tertulis di {OUT}/")
    print(f"  ground_truth.csv  {len(gt):,} baris")
    print(f"  submission.csv    {len(sub):,} baris")
    print(f"  RMSE contoh {np.sqrt(np.mean(err**2)):.4f} m · MAE {np.mean(np.abs(err)):.4f} m")


if __name__ == "__main__":
    main()
