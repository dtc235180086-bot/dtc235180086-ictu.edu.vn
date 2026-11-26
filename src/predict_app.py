"""Chương trình thử nghiệm: dự đoán chi phí bảo hiểm y tế cho một người mới.

Giao diện:   python src/predict_app.py
Dòng lệnh:   python src/predict_app.py --age 45 --sex male --bmi 32.5 --children 1 --smoker yes --region southeast
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES  # noqa: E402  (joblib cần module này để nạp pipeline)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "best_model.joblib"
REGIONS = ["northeast", "northwest", "southeast", "southwest"]


def vn(x, d=0):
    """Số kiểu Việt Nam: 40.568 ; 0,921."""
    return f"{x:,.{d}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def load():
    if not MODEL.exists():
        sys.exit(f"Chưa có {MODEL}. Hãy chạy: python src/chapter2_models.py")
    card = json.loads((ROOT / "models" / "model_card.json").read_text(encoding="utf-8"))
    return joblib.load(MODEL), card


def predict(model, age, sex, bmi, children, smoker, region):
    row = pd.DataFrame([{"age": age, "sex": sex, "bmi": bmi, "children": children,
                         "smoker": smoker, "region": region}])[FEATURES]
    return float(model.predict(row)[0])


def validate(age, bmi, children):
    if not 18 <= age <= 64:
        raise ValueError("Tuổi phải trong khoảng 18–64 (phạm vi dữ liệu huấn luyện).")
    if not 15 <= bmi <= 55:
        raise ValueError("BMI phải trong khoảng 15–55.")
    if not 0 <= children <= 5:
        raise ValueError("Số con phải trong khoảng 0–5.")


def run_cli(a):
    model, card = load()
    validate(a.age, a.bmi, a.children)
    p = predict(model, a.age, a.sex, a.bmi, a.children, a.smoker, a.region)
    rmse = card["test_metrics"]["rmse"]
    print(f"Mô hình: {card['model']}")
    print(f"Chi phí dự đoán: {vn(p)} USD/năm  (sai số RMSE trên tập kiểm tra ≈ {vn(rmse)} USD)")


def run_gui():
    import tkinter as tk
    from tkinter import messagebox, ttk

    model, card = load()
    root = tk.Tk()
    root.title("Dự đoán chi phí bảo hiểm y tế – Nhóm 15")
    root.geometry("460x430")
    frm = ttk.Frame(root, padding=16)
    frm.pack(fill="both", expand=True)
    ttk.Label(frm, text="Nhập thông tin người được bảo hiểm", font=("Segoe UI", 12, "bold")).grid(
        row=0, column=0, columnspan=2, pady=(0, 12), sticky="w")

    v_age, v_bmi, v_child = tk.IntVar(value=35), tk.DoubleVar(value=28.0), tk.IntVar(value=1)
    v_sex, v_smoker, v_region = tk.StringVar(value="male"), tk.StringVar(value="no"), tk.StringVar(value="southeast")
    fields = [("Tuổi (18–64)", ttk.Spinbox(frm, from_=18, to=64, textvariable=v_age, width=10)),
              ("Giới tính", ttk.Combobox(frm, values=["male", "female"], textvariable=v_sex, state="readonly", width=12)),
              ("BMI", ttk.Spinbox(frm, from_=15, to=55, increment=0.5, textvariable=v_bmi, width=10)),
              ("Số con (0–5)", ttk.Spinbox(frm, from_=0, to=5, textvariable=v_child, width=10)),
              ("Hút thuốc", ttk.Combobox(frm, values=["no", "yes"], textvariable=v_smoker, state="readonly", width=12)),
              ("Khu vực", ttk.Combobox(frm, values=REGIONS, textvariable=v_region, state="readonly", width=12))]
    for i, (lab, w) in enumerate(fields, start=1):
        ttk.Label(frm, text=lab).grid(row=i, column=0, sticky="w", pady=4)
        w.grid(row=i, column=1, sticky="w", pady=4)

    out = ttk.Label(frm, text="", font=("Segoe UI", 14, "bold"), foreground="#2a78d6")
    out.grid(row=8, column=0, columnspan=2, pady=(14, 2), sticky="w")
    note = ttk.Label(frm, text=f"Mô hình: {card['model']}\nRMSE kiểm tra ≈ {vn(card['test_metrics']['rmse'])} USD, "
                               f"R² = {vn(card['test_metrics']['r2'], 3)}", foreground="#52514e")
    note.grid(row=9, column=0, columnspan=2, sticky="w")

    def on_predict():
        try:
            age, bmi, ch = int(v_age.get()), float(v_bmi.get()), int(v_child.get())
            validate(age, bmi, ch)
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("Dữ liệu không hợp lệ", str(e))
            return
        p = predict(model, age, v_sex.get(), bmi, ch, v_smoker.get(), v_region.get())
        out.config(text=f"Chi phí dự đoán: {vn(p)} USD/năm")

    ttk.Button(frm, text="Dự đoán", command=on_predict).grid(row=7, column=0, columnspan=2, pady=(12, 0), sticky="w")
    root.mainloop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--age", type=int)
    ap.add_argument("--sex", choices=["male", "female"], default="male")
    ap.add_argument("--bmi", type=float)
    ap.add_argument("--children", type=int, default=0)
    ap.add_argument("--smoker", choices=["yes", "no"], default="no")
    ap.add_argument("--region", choices=REGIONS, default="southeast")
    a = ap.parse_args()
    if a.age is not None and a.bmi is not None:
        run_cli(a)
    else:
        run_gui()
