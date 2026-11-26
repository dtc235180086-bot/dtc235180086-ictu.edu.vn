"""Kiểu biểu đồ dùng chung cho mọi hình trong báo cáo (màu cố định theo vai trò)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Bảng màu phân loại cố định thứ tự (đã kiểm tra an toàn cho người mù màu)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
)
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
GRAY = "#8a8985"
INK = "#0b0b0b"
INK2 = "#52514e"
SEQ_CMAP = "Blues"
DIV_CMAP = "RdBu_r"

# Màu theo thực thể: người hút thuốc luôn màu cam, không hút luôn màu xanh
SMOKER_COLORS = {"no": BLUE, "yes": ORANGE}


def apply():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK2,
        "axes.edgecolor": "#c9c8c3",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "legend.frameon": False,
        "lines.linewidth": 2,
    })


def save(fig, path):
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def vn(v, decimals=0):
    """Số có dấu cách mảnh phân tách hàng nghìn — không nhầm với dấu thập phân."""
    return f"{v:,.{decimals}f}".replace(",", " ")
