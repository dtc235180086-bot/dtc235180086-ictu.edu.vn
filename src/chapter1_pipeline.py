"""Chương 1: hồ sơ dữ liệu, khảo sát thống kê, xử lý chất lượng và chia tập.

Chạy:  python src/chapter1_pipeline.py [data/insurance.csv]
Ghi:   results/stats_ch1.json, results/figures/fig1_*.png, data/processed/{train,test}.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from scipy import stats as sstats
from sklearn.model_selection import train_test_split

import plot_style as ps
import matplotlib.pyplot as plt

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
NUM = ["age", "bmi", "children"]
CAT = ["sex", "smoker", "region"]


def iqr_upper(s):
    q1, q3 = s.quantile([0.25, 0.75])
    return float(q3 + 1.5 * (q3 - q1))


def fig_workflow(path):
    steps = ["Tải & kiểm tra\nnguồn\ndữ liệu", "Lập hồ sơ\ndữ liệu\ngốc", "Khảo sát\nthống kê &\ntương quan",
             "Phát hiện\nvấn đề\nchất lượng", "Xử lý:\nloại trùng,\nxét ngoại lai", "Chia\ntrain/test\n(phân tầng)",
             "Huấn luyện,\nso sánh\nmô hình", "Đánh giá\ntrên tập\nkiểm tra"]
    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.set_axis_off()
    w, gap = 1.35, 0.18
    for i, s in enumerate(steps):
        x = i * (w + gap)
        color = ps.BLUE if i < 6 else ps.ORANGE
        ax.add_patch(FancyBboxPatch((x, 0), w, 1, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="white", ec=color, lw=2))
        ax.text(x + w / 2, 1.0, str(i + 1), ha="center", va="center", fontsize=10, color="white",
                fontweight="bold", bbox=dict(boxstyle="circle", fc=color, ec=color))
        ax.text(x + w / 2, 0.45, s, ha="center", va="center", fontsize=8, color=ps.INK)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap, 0.5), xytext=(x + w, 0.5),
                        arrowprops=dict(arrowstyle="-|>", color=ps.GRAY, lw=1.2))
    ax.set_xlim(-0.1, len(steps) * (w + gap))
    ax.set_ylim(-0.15, 1.25)
    ax.text(0, -0.12, "Xanh: chuẩn bị dữ liệu (Chương 1)   ·   Cam: mô hình hóa và đánh giá (Chương 2–3)",
            fontsize=8, color=ps.INK2)
    ps.save(fig, path)


def main(csv_path):
    ps.apply()
    FIG.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(csv_path)
    st = {"n_rows": int(len(raw)), "n_cols": int(raw.shape[1]),
          "dtypes": {c: str(t) for c, t in raw.dtypes.items()},
          "missing_cells": int(raw.isna().sum().sum()), "total_cells": int(raw.size)}
    dup_mask = raw.duplicated(keep="first")
    st["n_duplicates"] = int(dup_mask.sum())
    st["duplicate_rows"] = raw[raw.duplicated(keep=False)].to_dict(orient="records")

    desc = raw[NUM + ["charges"]].describe().T
    st["describe"] = {c: {k: float(desc.loc[c, k]) for k in ["mean", "std", "min", "25%", "50%", "75%", "max"]}
                      for c in desc.index}
    st["skew_charges"] = float(sstats.skew(raw["charges"]))
    st["skew_log_charges"] = float(sstats.skew(np.log1p(raw["charges"])))
    st["obese_share"] = float((raw["bmi"] >= 30).mean())

    st["categorical"] = {}
    for c in CAT:
        g = raw.groupby(c)["charges"].agg(["count", "mean"]).sort_values("count", ascending=False)
        st["categorical"][c] = [{"group": k, "count": int(r["count"]), "share": float(r["count"] / len(raw)),
                                 "mean_charges": float(r["mean"])} for k, r in g.iterrows()]
    st["children_counts"] = {int(k): int(v) for k, v in raw["children"].value_counts().sort_index().items()}
    st["pearson"] = raw[NUM + ["charges"]].corr().round(4).to_dict()
    enc = raw.assign(sex=(raw.sex == "male").astype(int), smoker=(raw.smoker == "yes").astype(int))
    sp = enc[NUM + ["sex", "smoker", "charges"]].corr(method="spearman")
    st["spearman"] = sp.round(4).to_dict()

    st["outliers"] = {
        "bmi_upper": iqr_upper(raw.bmi), "bmi_n": int((raw.bmi > iqr_upper(raw.bmi)).sum()),
        "charges_upper": iqr_upper(raw.charges), "charges_n": int((raw.charges > iqr_upper(raw.charges)).sum()),
    }
    st["outliers"]["charges_outlier_smoker_share"] = float(
        (raw.loc[raw.charges > iqr_upper(raw.charges), "smoker"] == "yes").mean())
    # Tương tác smoker × béo phì: chi phí trung bình của 4 nhóm
    inter = raw.assign(obese=raw.bmi >= 30).groupby(["smoker", "obese"])["charges"].agg(["count", "mean"])
    st["smoker_obese"] = [{"smoker": s, "obese": bool(o), "count": int(r["count"]), "mean": float(r["mean"])}
                          for (s, o), r in inter.iterrows()]

    # Xử lý chất lượng: chỉ loại dòng trùng hoàn toàn
    df = raw.drop_duplicates(keep="first").reset_index(drop=True)
    st["after"] = {"n_rows": int(len(df)), "smoker_share": float((df.smoker == "yes").mean()),
                   "skew_log_charges": float(sstats.skew(np.log1p(df.charges)))}

    train, test = train_test_split(df, test_size=0.2, stratify=df["smoker"], random_state=SEED)
    assert len(set(train.index) & set(test.index)) == 0 and len(train) + len(test) == len(df)
    st["split"] = {"train": int(len(train)), "test": int(len(test)),
                   "train_smokers": int((train.smoker == "yes").sum()),
                   "test_smokers": int((test.smoker == "yes").sum())}
    train.to_csv(ROOT / "data" / "processed" / "train.csv", index=False)
    test.to_csv(ROOT / "data" / "processed" / "test.csv", index=False)

    # ---- Hình ----
    fig_workflow(FIG / "fig1_1_quy_trinh.png")

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    axs[0].hist(raw.charges, bins=40, color=ps.BLUE, ec="white", lw=0.6)
    axs[0].axvline(raw.charges.mean(), color=ps.ORANGE, ls="--", lw=1.5, label=f"Trung bình {ps.vn(raw.charges.mean())}")
    axs[0].axvline(raw.charges.median(), color=ps.INK2, ls=":", lw=1.5, label=f"Trung vị {ps.vn(raw.charges.median())}")
    axs[0].set(title="Chi phí (USD)", xlabel="charges", ylabel="Số người")
    axs[0].legend(fontsize=8)
    axs[1].hist(np.log1p(raw.charges), bins=40, color=ps.AQUA, ec="white", lw=0.6)
    axs[1].set(title="log(1 + chi phí)", xlabel="log1p(charges)", ylabel="Số người")
    axs[0].text(0.98, 0.6, f"skew = {st['skew_charges']:.2f}", transform=axs[0].transAxes, ha="right", color=ps.INK2)
    axs[1].text(0.02, 0.9, f"skew = {st['skew_log_charges']:.2f}", transform=axs[1].transAxes, color=ps.INK2)
    ps.save(fig, FIG / "fig1_2_phan_bo_chi_phi.png")

    fig, axs = plt.subplots(1, 4, figsize=(12, 3.0))
    for ax, c in zip(axs, ["sex", "smoker", "region", "children"]):
        vc = raw[c].value_counts().sort_index()
        colors = [ps.SMOKER_COLORS.get(k, ps.BLUE) for k in vc.index] if c == "smoker" else ps.BLUE
        bars = ax.bar(vc.index.astype(str), vc.values, color=colors, width=0.7)
        ax.bar_label(bars, fontsize=8, color=ps.INK2)
        ax.set_title(c)
        ax.tick_params(axis="x", labelrotation=30 if c == "region" else 0)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("Số người")
    ps.save(fig, FIG / "fig1_3_bien_phan_loai.png")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for s in ["no", "yes"]:
        d = raw[raw.smoker == s]
        lab = "Hút thuốc" if s == "yes" else "Không hút thuốc"
        axs[0].scatter(d.age, d.charges, s=12, alpha=0.65, color=ps.SMOKER_COLORS[s], label=lab, lw=0)
        axs[1].scatter(d.bmi, d.charges, s=12, alpha=0.65, color=ps.SMOKER_COLORS[s], label=lab, lw=0)
    axs[1].axvline(30, color=ps.INK2, ls="--", lw=1)
    axs[1].text(30.4, 60000, "BMI = 30 (béo phì)", fontsize=8, color=ps.INK2)
    axs[0].set(title="Tuổi và chi phí", xlabel="age", ylabel="charges (USD)")
    axs[1].set(title="BMI và chi phí", xlabel="bmi")
    axs[0].legend(loc="upper left")
    ps.save(fig, FIG / "fig1_4_age_bmi_charges.png")

    fig, ax = plt.subplots(figsize=(6, 3.8))
    data = [raw.loc[raw.smoker == s, "charges"] for s in ["no", "yes"]]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, medianprops=dict(color=ps.INK))
    for patch, s in zip(bp["boxes"], ["no", "yes"]):
        patch.set(facecolor=ps.SMOKER_COLORS[s], alpha=0.75, edgecolor=ps.INK2)
    ax.set_xticks([1, 2], ["Không hút thuốc", "Hút thuốc"])
    ax.set(ylabel="charges (USD)", title="Chi phí theo tình trạng hút thuốc")
    ps.save(fig, FIG / "fig1_5_box_smoker.png")

    fig, ax = plt.subplots(figsize=(6.2, 5))
    labels = list(sp.columns)
    im = ax.imshow(sp.values, cmap=ps.DIV_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=35)
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = sp.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else ps.INK)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Ma trận tương quan Spearman")
    ps.save(fig, FIG / "fig1_6_spearman.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    groups = [("no", False), ("no", True), ("yes", False), ("yes", True)]
    vals = {(r["smoker"], r["obese"]): r["mean"] for r in st["smoker_obese"]}
    names = ["Không hút\nBMI<30", "Không hút\nBMI≥30", "Hút thuốc\nBMI<30", "Hút thuốc\nBMI≥30"]
    bars = ax.bar(names, [vals[g] for g in groups], color=[ps.SMOKER_COLORS[g[0]] for g in groups],
                  width=0.6)
    for b, g in zip(bars, groups):
        b.set_alpha(1 if g[1] else 0.55)
    ax.bar_label(bars, labels=[f"{ps.vn(vals[g])}" for g in groups], fontsize=8, color=ps.INK2)
    ax.set(ylabel="Chi phí trung bình (USD)", title="Tương tác smoker × béo phì")
    ax.grid(axis="x", visible=False)
    ps.save(fig, FIG / "fig1_7_smoker_x_obese.png")

    out = ROOT / "results" / "stats_ch1.json"
    out.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã đọc {st['n_rows']} dòng; trùng {st['n_duplicates']}; sau xử lý {st['after']['n_rows']} dòng")
    print(f"Train {st['split']['train']} ({st['split']['train_smokers']} hút thuốc), "
          f"test {st['split']['test']} ({st['split']['test_smokers']} hút thuốc)")
    print(f"Ngoại lai charges: {st['outliers']['charges_n']}, bmi: {st['outliers']['bmi_n']}")
    print(f"Ghi {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "insurance.csv")
