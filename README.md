# 📈 TMA Forecast Evaluator

Aplikasi evaluasi prediksi **time series tinggi muka air (TMA)** pada 30 pos pemantauan.
Unggah submission, aplikasi menghitung RMSE & MAE keseluruhan dan per pos.

---

## Struktur folder

```
tma_evaluator/
├── app.py                 # Aplikasi Streamlit (UI)
├── core.py                # Logika: load, validasi, merge, metrik, split
├── plots.py               # Figure Plotly
├── make_sample_data.py    # Generator contoh data
├── requirements.txt
├── README.md
└── sample_data/
    ├── ground_truth.csv   # Dipakai otomatis bila GT tidak diunggah
    └── submission.csv
```

## Cara menjalankan (lokal)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

Prasyarat: akun GitHub dan akun di [share.streamlit.io](https://share.streamlit.io).

1. **Push folder `tma_evaluator/` ke repo GitHub** (root repo = folder ini):

   ```bash
   git init
   git add .
   git commit -m "TMA Forecast Evaluator"
   git branch -M main
   git remote add origin https://github.com/<username>/<repo>.git
   git push -u origin main
   ```

   `.gitignore` sudah mengecualikan `.venv/` (600+ MB) dan cache — jangan sampai ikut.

2. Buka [share.streamlit.io](https://share.streamlit.io) → **New app** → **Deploy from GitHub**.
3. Isi: **Repository** = repo Anda, **Branch** = `main`, **Main file path** = `app.py`.
4. **Advanced settings** → pilih **Python 3.11** (versi yang teruji; `requirements.txt`
   memakai pandas 3.0 / numpy 2.4 yang butuh Python ≥ 3.11).
5. Klik **Deploy**. Streamlit membaca `requirements.txt` dan `.streamlit/config.toml`
   otomatis, lalu memberi URL publik.

> **Penting.** `sample_data/ground_truth.csv` **harus ikut ter-commit** — itulah ground
> truth default yang dipakai saat pengguna tidak mengunggah GT sendiri. Sudah termasuk
> dalam daftar commit dan tidak dikecualikan `.gitignore`.
>
> Tidak perlu `packages.txt` (tidak ada dependensi sistem) maupun `runtime.txt`
> (versi Python dipilih di Advanced settings, bukan dari file).

Membuat ulang contoh data (memakai `../test.csv` bila ada, selain itu data sintetis):

```bash
python make_sample_data.py
```

---

## Fitur

**Upload.** Ground truth **opsional** — bila tidak diunggah, aplikasi memakai
`sample_data/ground_truth.csv`. Submission wajib diunggah.

**Deteksi kolom otomatis** — tanpa konfigurasi manual. Identifier = kolom `id`
(atau kolom pertama); nilai = kolom `tma_mdpl` (atau kolom numerik terakhir).
Waktu & nama pos diambil dari kolom `datetime`/`nama_pos` bila ada, selain itu
diturunkan dari composite id.

**Validasi.** Evaluasi dihentikan bila ada identifier duplikat, nilai prediksi
kosong/nonnumerik/`inf`, identifier ground truth yang hilang, atau tidak ada
identifier yang cocok. Identifier berlebih hanya menjadi peringatan (diabaikan).

**Metrik.** RMSE, MAE, jumlah data, mean & std aktual/prediksi — keseluruhan
maupun per pos. Tabel per pos menyorot error tertinggi (merah) dan terendah
(hijau), dan dapat diunduh sebagai CSV.

**Visualisasi (4 tab).** Overview (scatter aktual vs prediksi + distribusi
residual), Time Series (aktual vs prediksi per pos, maks. 4 panel, dengan zoom,
pan, dan range slider), Per Pos (tabel + bar chart RMSE/MAE), Error Analysis
(10 absolute error terbesar).

```
RMSE = sqrt(mean((aktual - prediksi)^2))
MAE  = mean(abs(aktual - prediksi))
residual = aktual - prediksi
```

---

## Dua mode evaluasi

| Mode | Cara kerja |
|---|---|
| **Full Analysis** | Memakai **seluruh** ground truth untuk evaluasi. |
| **Kaggle-Style** | Split **kronologis global** 30/70 — lihat di bawah. |

### Split kronologis (Kaggle-Style)

Seluruh data diurutkan berdasarkan **waktu**, lalu:

- **30% baris paling awal → public**
- **70% sisanya → private**

Pembagian bersifat **global**, memakai satu batas waktu untuk seluruh data —
**bukan per pos**. Karena semua pos memiliki rentang waktu yang sama, ke-30 pos
tetap muncul di public maupun private. Tidak ada baris yang masuk ke kedua bagian.

Pada data contoh: public = 6.534 baris (s.d. 30 Nov 2025 12:00), private = 15.246
baris sesudahnya.

> Catatan: pemotongan dilakukan pada **jumlah baris** (tepat 30%), sehingga satu
> timestamp di batas dapat terbagi antara public dan private. Barisnya tetap unik
> dan tidak tumpang tindih.

**Private score** dapat disetel **Sembunyikan** (default) atau **Tampilkan** di
sidebar. Saat disembunyikan, nilai aktual private, RMSE/MAE private, dan seluruh
chart private tidak ditampilkan — hanya keterangan bahwa 70% data dipakai sebagai
private evaluation.

---

## Catatan format data

`nama_pos` dapat mengandung ` - ` (mis. `Arjowinangun - Pacitan`), sehingga
composite id dipecah **hanya pada pemisah pertama**:

```
2025-09-19 06:00:00 - Arjowinangun - Pacitan
└─ waktu ───────────┘ └─ pos ───────────────┘
```

Memecah pada semua pemisah akan merusak nama pos (30 pos terbaca menjadi 29).

## Cara membaca metrik

- **RMSE** memberi penalti lebih besar terhadap error besar (error dikuadratkan).
- **MAE** menunjukkan rata-rata absolute error apa adanya.
- Keduanya: **makin kecil makin baik**. RMSE ≫ MAE menandakan adanya outlier.
- Public memakai 30% ground truth, private 70% — **performa public yang bagus
  tidak selalu menjamin performa private yang bagus**.
