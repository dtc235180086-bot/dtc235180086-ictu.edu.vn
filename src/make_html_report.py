"""Sinh báo cáo trực quan một trang (results/BaoCao_TrucQuan.html) từ kết quả đã chạy.

Ảnh được nhúng base64 nên tệp HTML mở được ở mọi máy, không cần mạng.
"""
import base64
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def vn(x, d=0):
    s = f"{x:,.{d}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def img(name, caption):
    b = base64.b64encode((RES / "figures" / name).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'


def table(headers, rows, num_cols=()):
    th = "".join(f"<th{' class=n' if i in num_cols else ''}>{h}</th>" for i, h in enumerate(headers))
    trs = "".join("<tr>" + "".join(f"<td{' class=n' if i in num_cols else ''}>{c}</td>" for i, c in enumerate(r))
                  + "</tr>" for r in rows)
    return f"<div class=tw><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>"


def main():
    s1 = json.loads((RES / "stats_ch1.json").read_text(encoding="utf-8"))
    s2 = json.loads((RES / "metrics_ch2.json").read_text(encoding="utf-8"))
    bt = s2["best_test"]
    best = s2["best_model"]
    cv_rows = sorted(s2["cv"], key=lambda r: r["rmse_mean"])
    cv_tab = table(["Mô hình", "RMSE CV (USD)", "MAE CV (USD)", "R² CV", "RMSE test", "R² test"],
                   [[("<b>" + r["model"] + "</b>") if r["model"] == best else r["model"],
                     f'{vn(r["rmse_mean"])} ± {vn(r["rmse_std"])}', vn(r["mae_mean"]), vn(r["r2_mean"], 3),
                     vn(s2["test"][r["model"]]["rmse"]), vn(s2["test"][r["model"]]["r2"], 3)] for r in cv_rows],
                   num_cols=(1, 2, 3, 4, 5))
    smk = s2["best_by_smoker"]
    grp_tab = table(["Nhóm", "Số người (test)", "RMSE", "MAE", "R²"],
                    [["Không hút thuốc", smk["no"]["n"], vn(smk["no"]["rmse"]), vn(smk["no"]["mae"]), vn(smk["no"]["r2"], 3)],
                     ["Hút thuốc", smk["yes"]["n"], vn(smk["yes"]["rmse"]), vn(smk["yes"]["mae"]), vn(smk["yes"]["r2"], 3)]],
                    num_cols=(1, 2, 3, 4))
    kpis = [("RMSE kiểm tra", f'{vn(bt["rmse"])} USD', f'KTC 95%: {vn(bt["rmse_ci95"][0])}–{vn(bt["rmse_ci95"][1])}'),
            ("MAE kiểm tra", f'{vn(bt["mae"])} USD', "sai số tuyệt đối trung bình"),
            ("R² kiểm tra", vn(bt["r2"], 3), "tỷ lệ phương sai được giải thích"),
            ("Mốc dự đoán trung bình", f'{vn(s2["test"]["Dummy (trung bình)"]["rmse"])} USD', "RMSE của mô hình cơ sở")]
    kpi_html = "".join(f"<div class=kpi><div class=kl>{a}</div><div class=kv>{b}</div><div class=ks>{c}</div></div>"
                       for a, b, c in kpis)
    page = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Dự đoán chi phí bảo hiểm</title>
<style>
:root{{--bg:#fcfcfb;--card:#fff;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e0;--acc:#2a78d6;--acc2:#eb6834}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--card:#232322;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a38;--acc:#3987e5;--acc2:#d95926}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 "Segoe UI",system-ui,sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:24px 16px 64px}}header{{border-bottom:3px solid var(--acc);padding-bottom:16px;margin-bottom:24px}}
.org{{color:var(--ink2);font-size:13px;letter-spacing:.04em;text-transform:uppercase}}h1{{font-size:26px;line-height:1.3;margin:6px 0}}
h2{{font-size:20px;margin:36px 0 8px;padding-top:8px;border-top:1px solid var(--line)}}.meta{{color:var(--ink2)}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:20px 0}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}}.kl{{color:var(--ink2);font-size:13px}}
.kv{{font-size:24px;font-weight:700;color:var(--acc)}}.ks{{color:var(--ink2);font-size:12px}}
figure{{margin:16px 0;background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px}}figure img{{width:100%;height:auto;display:block}}
figcaption{{color:#52514e;font-size:13px;text-align:center;margin-top:6px}}.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}}
.tw{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;background:var(--card);font-size:14px}}th,td{{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left}}
th{{color:var(--ink2);font-weight:600}}.n{{text-align:right;font-variant-numeric:tabular-nums}}.note{{background:var(--card);border-left:4px solid var(--acc2);padding:10px 14px;border-radius:6px}}
</style></head><body><main>
<header><div class=org>Trường ĐH Công nghệ Thông tin và Truyền thông – Khoa Công nghệ Thông tin · Học phần Học máy</div>
<h1>Dự đoán chi phí bảo hiểm y tế cá nhân dựa trên đặc điểm nhân khẩu học và tình trạng sức khỏe</h1>
<div class=meta>Sinh viên thực hiện: <b>Ma Văn Hùng</b> · Giáo viên hướng dẫn: <b>TS. Nguyễn Tuấn Anh</b> · Thái Nguyên, 2026</div></header>
<p>Bộ dữ liệu Medical Cost Personal Datasets ({vn(s1["n_rows"])} bản ghi, 7 cột). Sau khi loại {s1["n_duplicates"]} dòng trùng còn
{vn(s1["after"]["n_rows"])} dòng, chia 80/20 phân tầng theo <i>smoker</i>: {vn(s1["split"]["train"])} dòng huấn luyện,
{s1["split"]["test"]} dòng kiểm tra. Mô hình cuối được chọn theo RMSE kiểm định chéo 5-fold: <b>{best}</b>.</p>
<div class=kpis>{kpi_html}</div>
<h2>1. Dữ liệu và khảo sát</h2>
{img("fig1_1_quy_trinh.png", "Quy trình thực hiện")}
<div class=grid2>{img("fig1_2_phan_bo_chi_phi.png", "Phân bố chi phí và log(1 + chi phí)")}
{img("fig1_5_box_smoker.png", "Chi phí theo tình trạng hút thuốc")}</div>
{img("fig1_4_age_bmi_charges.png", "Tuổi, BMI và chi phí – tô màu theo smoker")}
<div class=grid2>{img("fig1_7_smoker_x_obese.png", "Tương tác smoker × béo phì")}{img("fig1_6_spearman.png", "Tương quan Spearman")}</div>
<h2>2. So sánh mô hình</h2>
{img("fig2_1_cv_rmse.png", "RMSE kiểm định chéo 5-fold trên tập huấn luyện (thanh lỗi: độ lệch chuẩn giữa các fold)")}
{cv_tab}
<div class=grid2>{img("fig2_2_grid_gb.png", "Lưới siêu tham số Gradient Boosting")}{img("fig2_3_learning_curve.png", "Đường cong học")}</div>
<h2>3. Đánh giá trên tập kiểm tra</h2>
<div class=grid2>{img("fig3_1_actual_vs_pred.png", "Thực tế và dự đoán")}{img("fig3_2_residuals.png", "Phần dư")}</div>
{grp_tab}
<div class=grid2>{img("fig3_3_perm_importance.png", "Độ quan trọng hoán vị")}{img("fig3_4_mae_by_band.png", "MAE theo mức chi phí")}</div>
<div class=grid2>{img("fig3_5_whatif_bmi.png", "Chi phí dự đoán theo BMI")}
{img("fig3_7_giao_dien.png", "Giao diện chương trình thử nghiệm") if (RES / "figures" / "fig3_7_giao_dien.png").exists() else ""}</div>
<p class=note>Kết quả chỉ minh họa phương pháp trên bộ dữ liệu giáo dục; không dùng để định giá bảo hiểm thực tế.
Mọi số liệu trong trang được sinh tự động bởi <code>src/chapter1_pipeline.py</code> và <code>src/chapter2_models.py</code> (seed 42).</p>
</main></body></html>"""
    out = RES / "BaoCao_TrucQuan.html"
    out.write_text(page, encoding="utf-8")
    print(f"Đã ghi {out}")


if __name__ == "__main__":
    main()
