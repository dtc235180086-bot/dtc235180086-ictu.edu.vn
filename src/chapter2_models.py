"""Chương 2–3: so sánh mô hình bằng CV 5-fold, tối ưu siêu tham số, đánh giá trên tập kiểm tra.

Cần chạy chapter1_pipeline.py trước (tạo data/processed/train.csv, test.csv).
Chạy:  python src/chapter2_models.py
Ghi:   results/metrics_ch2.json, results/cv_results.csv, results/test_predictions.csv,
       results/figures/fig2_*.png, fig3_*.png, models/best_model.joblib
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, learning_curve

import plot_style as ps
import matplotlib.pyplot as plt
from features import FEATURES, make_pipeline

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
CV = KFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def test_metrics(y, p):
    return {"rmse": rmse(y, p), "mae": float(mean_absolute_error(y, p)), "r2": float(r2_score(y, p)),
            "mape": float(mean_absolute_percentage_error(y, p))}


def candidates():
    log_lr = TransformedTargetRegressor(LinearRegression(), func=np.log1p, inverse_func=np.expm1)
    return {
        "Dummy (trung bình)": make_pipeline(DummyRegressor()),
        "Hồi quy tuyến tính": make_pipeline(LinearRegression()),
        "Tuyến tính + nhãn log": make_pipeline(log_lr),
        "Ridge (α=1)": make_pipeline(Ridge(alpha=1.0)),
        "Tuyến tính + đặc trưng mới": make_pipeline(LinearRegression(), engineered=True),
        "Random Forest": make_pipeline(RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)),
        "Gradient Boosting": make_pipeline(GradientBoostingRegressor(random_state=SEED)),
    }


def run_cv(name, pipe, X, y):
    t = time.perf_counter()
    r = cross_validate(pipe, X, y, cv=CV, scoring=SCORING, n_jobs=-1)
    out = {"model": name, "time_s": time.perf_counter() - t}
    for k in SCORING:
        v = r[f"test_{k}"] * (-1 if k != "r2" else 1)
        out[f"{k}_mean"], out[f"{k}_std"] = float(v.mean()), float(v.std())
    return out


def main():
    ps.apply()
    FIG.mkdir(parents=True, exist_ok=True)
    (ROOT / "models").mkdir(exist_ok=True)
    train = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    Xtr, ytr, Xte, yte = train[FEATURES], train["charges"], test[FEATURES], test["charges"]
    res = {"n_train": len(train), "n_test": len(test)}

    # 1) So sánh mô hình cơ sở bằng CV 5-fold trên tập huấn luyện
    rows = []
    for name, pipe in candidates().items():
        rows.append(run_cv(name, pipe, Xtr, ytr))
        print(f"CV {name:30s} RMSE {rows[-1]['rmse_mean']:9.1f} ± {rows[-1]['rmse_std']:7.1f}  "
              f"R² {rows[-1]['r2_mean']:.4f}")

    # 2) Tối ưu siêu tham số cho hai mô hình cây
    gb_grid = {"model__n_estimators": [100, 200, 400], "model__learning_rate": [0.03, 0.05, 0.1],
               "model__max_depth": [2, 3, 4], "model__subsample": [0.8, 1.0],
               "model__min_samples_leaf": [1, 5, 10]}
    rf_grid = {"model__max_depth": [None, 4, 5, 6, 8], "model__min_samples_leaf": [1, 5, 10, 20],
               "model__max_features": [1.0, 0.6]}
    searches = {}
    for name, pipe, grid in [
        ("Gradient Boosting (tối ưu)", make_pipeline(GradientBoostingRegressor(random_state=SEED)), gb_grid),
        ("Random Forest (tối ưu)", make_pipeline(RandomForestRegressor(n_estimators=300, random_state=SEED)), rf_grid),
    ]:
        t = time.perf_counter()
        gs = GridSearchCV(pipe, grid, cv=CV, scoring=SCORING, refit="rmse", n_jobs=-1)
        gs.fit(Xtr, ytr)
        i = gs.best_index_
        cvr = gs.cv_results_
        rows.append({"model": name, "time_s": time.perf_counter() - t,
                     "rmse_mean": float(-cvr["mean_test_rmse"][i]), "rmse_std": float(cvr["std_test_rmse"][i]),
                     "mae_mean": float(-cvr["mean_test_mae"][i]), "mae_std": float(cvr["std_test_mae"][i]),
                     "r2_mean": float(cvr["mean_test_r2"][i]), "r2_std": float(cvr["std_test_r2"][i])})
        searches[name] = gs
        res.setdefault("grid", {})[name] = {
            "n_configs": len(cvr["params"]),
            "best_params": {k.replace("model__", ""): v for k, v in gs.best_params_.items()},
            "time_s": rows[-1]["time_s"]}
        print(f"Grid {name}: {len(cvr['params'])} cấu hình, tốt nhất {gs.best_params_} "
              f"RMSE {rows[-1]['rmse_mean']:.1f}")

    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(ROOT / "results" / "cv_results.csv", index=False)
    res["cv"] = rows

    # 3) Huấn luyện lại mọi mô hình trên toàn bộ tập train, đánh giá một lần trên tập test
    fitted = {n: p.fit(Xtr, ytr) for n, p in candidates().items()}
    fitted.update({n: gs.best_estimator_ for n, gs in searches.items()})
    res["test"] = {n: test_metrics(yte, m.predict(Xte)) for n, m in fitted.items()}

    # Chọn mô hình cuối theo RMSE kiểm định chéo (không nhìn tập test)
    best_name = cv_df.sort_values("rmse_mean").iloc[0]["model"]
    best = fitted[best_name]
    res["best_model"] = best_name
    pred = best.predict(Xte)
    res["best_test"] = test_metrics(yte, pred)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, len(yte), len(yte))
        boots.append(rmse(yte.values[idx], pred[idx]))
    res["best_test"]["rmse_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    by_smoker = {}
    for s in ["no", "yes"]:
        m = (test.smoker == s).values
        by_smoker[s] = test_metrics(yte[m], pred[m]) | {"n": int(m.sum())}
    res["best_by_smoker"] = by_smoker
    print(f"Mô hình cuối: {best_name} → test {res['best_test']}")

    t = time.perf_counter()
    for _ in range(20):
        best.predict(Xte)
    res["predict_ms_per_row"] = (time.perf_counter() - t) / 20 / len(Xte) * 1000

    out_pred = test.copy()
    out_pred["predicted"] = pred
    out_pred["error"] = out_pred["predicted"] - out_pred["charges"]
    out_pred.to_csv(ROOT / "results" / "test_predictions.csv", index=False)
    res["worst_cases"] = out_pred.reindex(out_pred.error.abs().sort_values(ascending=False).index).head(5) \
        .round(2).to_dict(orient="records")

    joblib.dump(best, ROOT / "models" / "best_model.joblib")
    (ROOT / "models" / "model_card.json").write_text(json.dumps({
        "model": best_name, "features": FEATURES, "target": "charges (USD)", "seed": SEED,
        "train_rows": len(train), "test_metrics": res["best_test"],
        "params": res.get("grid", {}).get(best_name, {}).get("best_params")}, ensure_ascii=False, indent=2,
        default=str), encoding="utf-8")

    # Hệ số mô hình tuyến tính có đặc trưng mới để giải thích
    lin = fitted["Tuyến tính + đặc trưng mới"]
    names = lin.named_steps["prep"].get_feature_names_out()
    coefs = lin.named_steps["model"].coef_
    res["linear_fe_coefs"] = dict(zip(names, map(float, coefs)))

    pi = permutation_importance(best, Xte, yte, scoring="neg_root_mean_squared_error", n_repeats=30,
                                random_state=SEED, n_jobs=-1)
    res["perm_importance"] = {f: {"mean": float(m), "std": float(s)}
                              for f, m, s in zip(FEATURES, pi.importances_mean, pi.importances_std)}

    sizes, tr_sc, va_sc = learning_curve(best, Xtr, ytr, cv=CV, scoring="neg_root_mean_squared_error",
                                         train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1)
    res["learning_curve"] = {"sizes": sizes.tolist(), "train_rmse": (-tr_sc.mean(1)).tolist(),
                             "val_rmse": (-va_sc.mean(1)).tolist()}

    # ---- Hình ----
    d = cv_df.sort_values("rmse_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = [ps.ORANGE if m == best_name else ps.BLUE for m in d.model]
    bars = ax.barh(d.model, d.rmse_mean, xerr=d.rmse_std, color=colors, height=0.6,
                   error_kw=dict(ecolor=ps.INK2, lw=1, capsize=3))
    ax.bar_label(bars, labels=[f"{ps.vn(v)}" for v in d.rmse_mean], padding=22, fontsize=8, color=ps.INK2)
    ax.set(xlabel="RMSE kiểm định chéo 5-fold (USD, thấp hơn là tốt hơn)",
           title="So sánh mô hình trên tập huấn luyện")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, d.rmse_mean.max() * 1.25)
    ps.save(fig, FIG / "fig2_1_cv_rmse.png")

    gs = searches["Gradient Boosting (tối ưu)"]
    cvr = pd.DataFrame(gs.cv_results_)
    piv = cvr.assign(lr=cvr.param_model__learning_rate.astype(float), depth=cvr.param_model__max_depth.astype(int),
                     rmse=-cvr.mean_test_rmse).groupby(["depth", "lr"]).rmse.min().unstack()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    im = ax.imshow(piv.values, cmap=ps.SEQ_CMAP + "_r", aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [str(c) for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), [str(i) for i in piv.index])
    ax.set(xlabel="learning_rate", ylabel="max_depth", title="Gradient Boosting: RMSE CV tốt nhất")
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{ps.vn(piv.values[i, j])}", ha="center", va="center", fontsize=9, color=ps.INK)
    fig.colorbar(im, ax=ax, shrink=0.85)
    ps.save(fig, FIG / "fig2_2_grid_gb.png")

    lc = res["learning_curve"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(lc["sizes"], lc["train_rmse"], "-o", color=ps.BLUE, ms=5, label="RMSE huấn luyện")
    ax.plot(lc["sizes"], lc["val_rmse"], "-o", color=ps.ORANGE, ms=5, label="RMSE kiểm định")
    ax.set(xlabel="Số mẫu huấn luyện", ylabel="RMSE (USD)", title=f"Đường cong học – {best_name}")
    ax.legend()
    ps.save(fig, FIG / "fig2_3_learning_curve.png")

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for s in ["no", "yes"]:
        m = (test.smoker == s).values
        ax.scatter(yte[m], pred[m], s=16, alpha=0.75, color=ps.SMOKER_COLORS[s], lw=0,
                   label="Hút thuốc" if s == "yes" else "Không hút thuốc")
    lim = [0, max(yte.max(), pred.max()) * 1.05]
    ax.plot(lim, lim, "--", color=ps.GRAY, lw=1, label="Dự đoán = thực tế")
    ax.set(xlim=lim, ylim=lim, xlabel="Chi phí thực tế (USD)", ylabel="Chi phí dự đoán (USD)",
           title="Thực tế và dự đoán trên tập kiểm tra")
    ax.legend(loc="upper left")
    ps.save(fig, FIG / "fig3_1_actual_vs_pred.png")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for s in ["no", "yes"]:
        m = (test.smoker == s).values
        ax.scatter(pred[m], (yte - pred)[m], s=16, alpha=0.75, color=ps.SMOKER_COLORS[s], lw=0,
                   label="Hút thuốc" if s == "yes" else "Không hút thuốc")
    ax.axhline(0, color=ps.GRAY, lw=1)
    ax.set(xlabel="Chi phí dự đoán (USD)", ylabel="Phần dư = thực tế − dự đoán", title="Phần dư trên tập kiểm tra")
    ax.legend()
    ps.save(fig, FIG / "fig3_2_residuals.png")

    imp = pd.Series({k: v["mean"] for k, v in res["perm_importance"].items()}).sort_values()
    err = pd.Series({k: v["std"] for k, v in res["perm_importance"].items()})[imp.index]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars = ax.barh(imp.index, imp.values, xerr=err.values, color=ps.BLUE, height=0.6,
                   error_kw=dict(ecolor=ps.INK2, lw=1, capsize=3))
    ax.bar_label(bars, labels=[f"+{ps.vn(v)}" for v in imp.values], padding=14, fontsize=8, color=ps.INK2)
    ax.set(xlabel="RMSE tăng thêm khi xáo trộn đặc trưng (USD)", title="Độ quan trọng hoán vị")
    ax.set_xlim(0, imp.max() * 1.3)
    ax.grid(axis="y", visible=False)
    ps.save(fig, FIG / "fig3_3_perm_importance.png")

    # Sai số theo mức chi phí
    bands = pd.qcut(yte, 5, labels=["Q1 (thấp)", "Q2", "Q3", "Q4", "Q5 (cao)"])
    band_mae = pd.Series(np.abs(yte - pred)).groupby(bands, observed=True).mean()
    res["mae_by_band"] = {str(k): float(v) for k, v in band_mae.items()}
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars = ax.bar(band_mae.index.astype(str), band_mae.values, color=ps.BLUE, width=0.6)
    ax.bar_label(bars, labels=[f"{ps.vn(v)}" for v in band_mae.values], fontsize=8, color=ps.INK2)
    ax.set(ylabel="MAE (USD)", xlabel="Nhóm chi phí thực tế (ngũ phân vị)", title="Sai số tuyệt đối theo mức chi phí")
    ax.grid(axis="x", visible=False)
    ps.save(fig, FIG / "fig3_4_mae_by_band.png")

    # Đường "nếu-thì": chi phí dự đoán theo BMI cho người 45 tuổi
    grid = np.linspace(18, 45, 60)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    whatif = {}
    for s in ["no", "yes"]:
        q = pd.DataFrame({"age": 45, "sex": "male", "bmi": grid, "children": 1, "smoker": s, "region": "southeast"})
        p = best.predict(q[FEATURES])
        whatif[s] = p.tolist()
        ax.plot(grid, p, color=ps.SMOKER_COLORS[s], label="Hút thuốc" if s == "yes" else "Không hút thuốc")
    ax.axvline(30, color=ps.INK2, ls="--", lw=1)
    ax.set(xlabel="BMI", ylabel="Chi phí dự đoán (USD)", title="Mô hình học được tương tác smoker × BMI (nam, 45 tuổi)")
    ax.legend()
    res["whatif_bmi"] = {"bmi": grid.tolist(), **whatif}
    ps.save(fig, FIG / "fig3_5_whatif_bmi.png")

    # So sánh trên tập test (RMSE và R²)
    tdf = pd.DataFrame(res["test"]).T.sort_values("rmse", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = [ps.ORANGE if m == best_name else ps.BLUE for m in tdf.index]
    bars = ax.barh(tdf.index, tdf.rmse, color=colors, height=0.6)
    ax.bar_label(bars, labels=[f"{ps.vn(r)}  (R² {q:.3f})" for r, q in zip(tdf.rmse, tdf.r2)], padding=4,
                 fontsize=8, color=ps.INK2)
    ax.set(xlabel="RMSE trên tập kiểm tra (USD)", title=f"Kết quả trên tập kiểm tra độc lập ({len(test)} người)")
    ax.set_xlim(0, tdf.rmse.max() * 1.35)
    ax.grid(axis="y", visible=False)
    ps.save(fig, FIG / "fig3_6_test_rmse.png")

    (ROOT / "results" / "metrics_ch2.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str),
                                                       encoding="utf-8")
    print("Đã ghi results/metrics_ch2.json và models/best_model.joblib")


if __name__ == "__main__":
    main()
