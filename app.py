import os
import json
import math
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

MODEL, INFO, LOAD_ERROR = None, None, None
try:
    MODEL = joblib.load(os.path.join(BASE, "model.joblib"))
    with open(os.path.join(BASE, "metrics.json")) as f:
        INFO = json.load(f)
except FileNotFoundError:
    LOAD_ERROR = "File model.joblib / metrics.json belum ada. Jalankan `python train_model.py` terlebih dahulu."

MAX_MENIT = 1440                      # 24 jam
KAPASITAS = [3000, 4000, 5000, 6000]  # pilihan kapasitas baterai (mAh)


def angka(n, dec=0):
    """Format angka gaya Indonesia: 1.234,5"""
    s = f"{n:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def jam_menit(menit):
    total = int(round(menit))
    j, m = divmod(total, 60)
    if j and m:
        return f"{j} jam {m} menit"
    return f"{j} jam" if j else f"{m} menit"


app.jinja_env.filters["angka"] = angka
app.jinja_env.filters["jam_menit"] = jam_menit


def ringkas_model(info):
    """Terjemahkan angka-angka model ke kalimat sederhana."""
    r2 = info["r2"]
    if r2 >= 0.8:
        label, warna = "sangat baik", "success"
    elif r2 >= 0.6:
        label, warna = "cukup baik", "primary"
    elif r2 >= 0.4:
        label, warna = "sedang", "warning"
    else:
        label, warna = "lemah", "danger"
    return {
        "r2_persen": round(r2 * 100),
        "r2_label": label,
        "r2_warna": warna,
        "slope_jam": info["coef"] * 60,
        "intercept_wajar": info["intercept"] >= 0,
    }


def info_tampilan(info):
    x_min, x_max = math.floor(info["x_min"]), math.ceil(info["x_max"])
    presets = [p for p in (60, 120, 180, 240, 300, 420, 600) if x_min <= p <= x_max][:5]
    if not presets:
        presets = sorted({x_min, round((x_min + x_max) / 2), x_max})
    return {
        "x_min": x_min, "x_max": x_max, "presets": presets,
        "default_x": round(info.get("x_mean", (x_min + x_max) / 2)),
        "n_total": info.get("n_total", info["n_train"] + info["n_test"]),
    }


def tingkat(pred, info):
    q33, q66 = info.get("y_q33"), info.get("y_q66")
    if q33 is None:
        return None
    if pred <= q33:
        return ("Tergolong hemat", "success", "lebih hemat dari kebanyakan pengguna di data")
    if pred <= q66:
        return ("Tergolong sedang", "warning text-dark", "mirip kebanyakan pengguna di data")
    return ("Tergolong boros", "danger", "lebih boros dari kebanyakan pengguna di data")


@app.route("/", methods=["GET", "POST"])
def index():
    if LOAD_ERROR:
        return render_template("index.html", load_error=LOAD_ERROR)

    result, error, warning, raw = None, None, None, ""
    kap = 5000

    if request.method == "POST":
        raw = request.form.get("durasi", "").strip().replace(",", ".")
        try:
            kap = int(request.form.get("kapasitas", 5000))
        except ValueError:
            kap = 5000
        if kap not in KAPASITAS:
            kap = 5000
        try:
            durasi = float(raw)
            if math.isnan(durasi) or durasi <= 0 or durasi > MAX_MENIT:
                error = f"Durasi harus lebih dari 0 dan maksimal {MAX_MENIT} menit (24 jam)."
        except ValueError:
            error = "Masukkan angka saja, misalnya 180."

        if not error:
            pred = max(0.0, float(MODEL.predict(pd.DataFrame({"app_usage_min": [durasi]}))[0]))
            y_min, y_max = INFO.get("y_min", 0), INFO.get("y_max", pred * 2 or 1)
            pos = 0 if y_max == y_min else (pred - y_min) / (y_max - y_min) * 100
            result = {
                "durasi": durasi, "pred": pred,
                "low": max(0.0, pred - INFO["rmse"]), "high": pred + INFO["rmse"],
                "persen_baterai": pred / kap * 100, "kap": kap,
                "pos": min(100, max(0, pos)), "tingkat": tingkat(pred, INFO),
            }
            if durasi < INFO["x_min"] or durasi > INFO["x_max"]:
                warning = (f"Angka {angka(durasi)} menit berada di luar rentang data yang dipelajari model "
                           f"({angka(INFO['x_min'])} sampai {angka(INFO['x_max'])} menit). "
                           f"Hasilnya hanya tebakan kasar dan kurang bisa dipercaya.")

    return render_template("index.html", info=INFO, v=info_tampilan(INFO), m=ringkas_model(INFO),
                           result=result, error=error, warning=warning, raw=raw,
                           kapasitas=KAPASITAS, kap=kap, max_menit=MAX_MENIT, load_error=None)


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
