import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/user_behavior_dataset.csv"
COL_X = "App Usage Time (min/day)"
COL_Y = "Battery Drain (mAh/day)"

# 1. Muat data
df = pd.read_csv(CSV_PATH)
print("--- 5 Baris Pertama ---")
print(df.head())
print("\n--- Nama Kolom ---")
print(df.columns)
print("\n--- Info ---")
print(df.info())

# 2. Pra-pemrosesan: ambil 2 kolom, hapus missing value, duplikat, dan outlier (IQR)
data = df[[COL_X, COL_Y]].rename(columns={COL_X: "app_usage_min", COL_Y: "battery_drain_mah"})
n_awal = len(data)
data = data.dropna().drop_duplicates()
print(f"\nMissing/duplikat dibuang: {n_awal - len(data)} baris")

def iqr_mask(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return s.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

mask = iqr_mask(data["app_usage_min"]) & iqr_mask(data["battery_drain_mah"])
print(f"Outlier (IQR) dibuang: {(~mask).sum()} baris")
data = data[mask].reset_index(drop=True)
print(f"Jumlah data akhir: {len(data)}")
print("\nKorelasi X-y:", round(data["app_usage_min"].corr(data["battery_drain_mah"]), 4))

# 3. Split, latih, evaluasi
X = data[["app_usage_min"]]
y = data["battery_drain_mah"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)

mae = metrics.mean_absolute_error(y_test, pred)
rmse = float(np.sqrt(metrics.mean_squared_error(y_test, pred)))
r2 = metrics.r2_score(y_test, pred)
print(f"\nIntercept : {model.intercept_:.4f}")
print(f"Koefisien : {model.coef_[0]:.4f}")
print(f"MAE  : {mae:.2f}\nRMSE : {rmse:.2f}\nR2   : {r2:.4f}")

# 4. Scatterplot + garis regresi (validasi asumsi linear)
plt.figure(figsize=(7, 4.5))
plt.scatter(X_train, y_train, s=14, alpha=0.5, label="Data latih")
plt.scatter(X_test, y_test, s=14, alpha=0.8, color="orange", label="Data uji")
xs = np.linspace(X["app_usage_min"].min(), X["app_usage_min"].max(), 100)
plt.plot(xs, model.intercept_ + model.coef_[0] * xs, color="red", lw=2, label="Garis regresi")
plt.xlabel("Durasi Penggunaan Aplikasi (menit/hari)")
plt.ylabel("Konsumsi Baterai (mAh/hari)")
plt.title("Durasi Penggunaan Aplikasi vs Konsumsi Baterai")
plt.legend()
plt.tight_layout()
plt.savefig("static/scatter.png", dpi=130)

# 5. Simpan model & metadata
joblib.dump(model, "model.joblib")
info = {
    "intercept": float(model.intercept_),
    "coef": float(model.coef_[0]),
    "r2": float(r2), "rmse": rmse, "mae": float(mae),
    "x_min": float(X["app_usage_min"].min()), "x_max": float(X["app_usage_min"].max()),
    "n_train": int(len(X_train)), "n_test": int(len(X_test)), "n_total": int(len(data)),
    "y_min": float(y.min()), "y_max": float(y.max()),
    "y_q33": float(y.quantile(0.33)), "y_q66": float(y.quantile(0.66)),
    "x_mean": float(X["app_usage_min"].mean()), "y_mean": float(y.mean()),
    "corr": float(data["app_usage_min"].corr(y)),
}
with open("metrics.json", "w") as f:
    json.dump(info, f, indent=2)
print("\nTersimpan: model.joblib, metrics.json, static/scatter.png")
