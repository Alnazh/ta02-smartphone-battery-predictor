# Prediksi Konsumsi Baterai Smartphone (Regresi Linear Sederhana)

Aplikasi web Flask: memprediksi **konsumsi baterai (mAh/hari)** dari **durasi penggunaan aplikasi (menit/hari)**.
Dataset: [Mobile Device Usage and User Behavior Dataset (Kaggle)](https://www.kaggle.com/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset)

## Menjalankan di komputer sendiri
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Taruh user_behavior_dataset.csv (dari Kaggle) di folder data/
# 2. Latih model (menghasilkan model.joblib, metrics.json, static/scatter.png)
python train_model.py

# 3. Jalankan aplikasi
python app.py                    # buka http://127.0.0.1:5000
```

## Deploy (Render)
1. Push seluruh folder ke GitHub, **termasuk** `model.joblib`, `metrics.json`, dan `static/scatter.png` (jalankan `train_model.py` dulu).
2. Render: New > Web Service > pilih repo.
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn app:app`

## Struktur
- `train_model.py` : EDA ringkas, pembersihan, latih model, evaluasi (R2, RMSE, MAE), simpan dengan Joblib
- `app.py` : backend Flask (muat model, validasi input, prediksi)
- `templates/index.html` : antarmuka Bootstrap
- `data/` : tempat file CSV dataset
