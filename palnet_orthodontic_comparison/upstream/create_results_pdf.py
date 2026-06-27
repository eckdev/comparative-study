import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
OUT_DIR = REPO / "output" / "pdf"
OUT_PATH = OUT_DIR / "palnet_orthodontic_sonuclar_raporu.pdf"

RUN_RAW = ROOT / "runs" / "orthodontic_palnet_patch100_e40"
RUN_ALIGNED_OLD = ROOT / "runs" / "orthodontic_palnet_procrustes_rigid_patch100_e40"
RUN_SHORT = ROOT / "runs" / "orthodontic_palnet_procrustes_rigid_20260627_120534_patch100_e40"
RUN_LATEST = ROOT / "runs" / "orthodontic_palnet_procrustes_rigid_20260627_143801_patch1000_surface100k_e200"
TRANSFORM_LATEST = ROOT / "transforms" / "orthodontic_procrustes_rigid_20260627_143801"

LANDMARK_NAMES = [
    "Tr",
    "G",
    "N",
    "Prn",
    "Col",
    "Sn",
    "Phi",
    "Ls",
    "Stm",
    "Li",
    "Pg",
    "Gn",
    "Me",
    "Ex_R",
    "End_R",
    "End_L",
    "Ex_L",
    "Al_R",
    "Al_L",
    "Ch_R",
    "Ch_L",
    "Go_R",
    "Go_L",
]

ANATOMICAL_GROUPS = [
    ("Orta hat profil", [0, 1, 2, 10, 11, 12]),
    ("Burun ve filtrum", [3, 4, 5, 6, 17, 18]),
    ("Dudak ve ağız", [7, 8, 9, 19, 20]),
    ("Göz çevresi", [13, 14, 15, 16]),
    ("Mandibular açı", [21, 22]),
]


def register_fonts():
    font_path = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    bold_path = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    if font_path.exists():
        pdfmetrics.registerFont(TTFont("ReportFont", str(font_path)))
    else:
        pdfmetrics.registerFont(TTFont("ReportFont", "/Library/Fonts/Arial Unicode.ttf"))
    if bold_path.exists():
        pdfmetrics.registerFont(TTFont("ReportFont-Bold", str(bold_path)))
    else:
        pdfmetrics.registerFont(TTFont("ReportFont-Bold", str(font_path)))


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value, digits=3):
    return f"{float(value):.{digits}f}"


def load_group_metrics(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_sample_errors(path):
    grouped = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["sample_id"]].append(float(row["localization_error"]))
    values = []
    for sample_id, errors in grouped.items():
        values.append(
            {
                "sample_id": sample_id,
                "ale": sum(errors) / len(errors),
                "max": max(errors),
                "median": sorted(errors)[len(errors) // 2],
            }
        )
    return sorted(values, key=lambda item: item["ale"])


def load_landmark_stats(path):
    grouped = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[int(row["landmark"])].append(float(row["localization_error"]))

    rows = []
    for landmark in sorted(grouped):
        errors = grouped[landmark]
        rows.append(
            {
                "landmark": landmark,
                "name": LANDMARK_NAMES[landmark] if landmark < len(LANDMARK_NAMES) else f"P{landmark}",
                "mean": sum(errors) / len(errors),
                "median": statistics.median(errors),
                "std": statistics.pstdev(errors),
                "max": max(errors),
            }
        )
    return rows


def sample_stats(path, sample_id):
    grouped = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["sample_id"]].append((int(row["landmark"]), float(row["localization_error"])))
    values = grouped[sample_id]
    if not values:
        return None
    errors = [v for _, v in values]
    top_landmark, top_error = max(values, key=lambda item: item[1])
    return {
        "ale": sum(errors) / len(errors),
        "median": statistics.median(errors),
        "max": top_error,
        "top_landmark": top_landmark,
    }


def anatomical_group_stats(landmark_rows):
    by_index = {row["landmark"]: row for row in landmark_rows}
    rows = []
    for name, indices in ANATOMICAL_GROUPS:
        means = [by_index[i]["mean"] for i in indices]
        medians = [by_index[i]["median"] for i in indices]
        maxima = [by_index[i]["max"] for i in indices]
        rows.append(
            {
                "group": name,
                "landmarks": ", ".join(f"{i}-{by_index[i]['name']}" for i in indices),
                "mean": sum(means) / len(means),
                "median": sum(medians) / len(medians),
                "max": max(maxima),
            }
        )
    return rows


def paragraph(text, style):
    return Paragraph(text.replace("\n", "<br/>"), style)


def metric_cards(metrics):
    data = [
        [
            "Model çıktısı",
            "ALE",
            "Medyan",
            "Std",
        ],
        [
            "PAL-Net raw",
            fmt(metrics["palnet_raw"]["ale"]),
            fmt(metrics["palnet_raw"]["median"]),
            fmt(metrics["palnet_raw"]["std"]),
        ],
        [
            "PAL-Net snapped",
            fmt(metrics["palnet_snapped"]["ale"]),
            fmt(metrics["palnet_snapped"]["median"]),
            fmt(metrics["palnet_snapped"]["std"]),
        ],
        [
            "Mean-template baseline",
            fmt(metrics["mean_shape_baseline_snapped"]["ale"]),
            fmt(metrics["mean_shape_baseline_snapped"]["median"]),
            fmt(metrics["mean_shape_baseline_snapped"]["std"]),
        ],
    ]
    return styled_table(data, col_widths=[6.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm])


def styled_table(data, col_widths=None, font_size=8.8):
    table = Table(data, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#21313f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "ReportFont-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "ReportFont"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cad0d6")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7f9")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def bar_table(rows, label_key, value_key, max_value, width_cm=8.0):
    data = [["Grup", "ALE", "Göreli görünüm"]]
    for row in rows:
        value = float(row[value_key])
        bar_width = max(0.15, value / max_value * width_cm)
        data.append(
            [
                row[label_key],
                fmt(value),
                Table(
                    [["", ""]],
                    colWidths=[bar_width * cm, max(0.1, (width_cm - bar_width)) * cm],
                    rowHeights=[0.24 * cm],
                    style=TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#4f8fcf")),
                            ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#e7edf3")),
                            ("BOX", (0, 0), (-1, -1), 0.1, colors.HexColor("#d7dde3")),
                        ]
                    ),
                ),
            ]
        )
    return styled_table(data, col_widths=[4.2 * cm, 2.2 * cm, 8.6 * cm], font_size=8.3)


def color_interp(value, low, high):
    if high <= low:
        t = 0
    else:
        t = max(0, min(1, (value - low) / (high - low)))
    if t < 0.5:
        local = t / 0.5
        r = 0x2e + (0xff - 0x2e) * local
        g = 0xb8 + (0xd1 - 0xb8) * local
        b = 0x72 + (0x66 - 0x72) * local
    else:
        local = (t - 0.5) / 0.5
        r = 0xff + (0xe0 - 0xff) * local
        g = 0xd1 + (0x4f - 0xd1) * local
        b = 0x66 + (0x4f - 0x66) * local
    return colors.Color(r / 255, g / 255, b / 255)


def horizontal_bar_chart(rows, label_key, value_key, title, max_value=None, width=15.5 * cm, height=None):
    row_h = 0.58 * cm
    top = 0.85 * cm
    bottom = 0.35 * cm
    height = height or top + bottom + row_h * len(rows)
    drawing = Drawing(width, height)
    drawing.add(String(0, height - 0.35 * cm, title, fontName="ReportFont-Bold", fontSize=9.5, fillColor=colors.HexColor("#17212b")))
    max_value = max_value or max(float(row[value_key]) for row in rows)
    label_w = 4.2 * cm
    value_w = 1.4 * cm
    bar_w = width - label_w - value_w - 0.5 * cm
    for i, row in enumerate(rows):
        y = height - top - (i + 1) * row_h + 0.12 * cm
        value = float(row[value_key])
        drawing.add(String(0, y + 0.07 * cm, str(row[label_key]), fontName="ReportFont", fontSize=7.5, fillColor=colors.black))
        drawing.add(Rect(label_w, y, bar_w, 0.22 * cm, fillColor=colors.HexColor("#e7edf3"), strokeColor=None))
        drawing.add(Rect(label_w, y, max(0.05 * cm, bar_w * value / max_value), 0.22 * cm, fillColor=colors.HexColor("#4f8fcf"), strokeColor=None))
        drawing.add(String(label_w + bar_w + 0.25 * cm, y + 0.03 * cm, fmt(value), fontName="ReportFont", fontSize=7.5, fillColor=colors.black))
    return drawing


def comparison_chart(raw, aligned_old, short, latest):
    rows = [
        {"label": "Ham PLY", "ale": raw["palnet_snapped"]["ale"]},
        {"label": "Procrustes önceki", "ale": aligned_old["palnet_snapped"]["ale"]},
        {"label": "Kısa eğitim", "ale": short["palnet_snapped"]["ale"]},
        {"label": "Uzun eğitim", "ale": latest["palnet_snapped"]["ale"]},
    ]
    return horizontal_bar_chart(rows, "label", "ale", "Koşulara Göre Snapped ALE", max_value=max(row["ale"] for row in rows))


def landmark_heatmap(landmark_rows, width=15.4 * cm):
    cols = 6
    cell_w = width / cols
    cell_h = 1.25 * cm
    rows = 4
    title_h = 0.65 * cm
    legend_h = 0.55 * cm
    height = title_h + rows * cell_h + legend_h
    drawing = Drawing(width, height)
    drawing.add(String(0, height - 0.35 * cm, "Landmark Mean Hata Isı Haritası", fontName="ReportFont-Bold", fontSize=9.5, fillColor=colors.HexColor("#17212b")))
    low = min(row["mean"] for row in landmark_rows)
    high = max(row["mean"] for row in landmark_rows)
    for idx, row in enumerate(landmark_rows):
        col = idx % cols
        line = idx // cols
        x = col * cell_w
        y = height - title_h - (line + 1) * cell_h
        fill = color_interp(row["mean"], low, high)
        drawing.add(Rect(x + 0.05 * cm, y + 0.08 * cm, cell_w - 0.10 * cm, cell_h - 0.14 * cm, fillColor=fill, strokeColor=colors.white, strokeWidth=1))
        drawing.add(String(x + 0.15 * cm, y + 0.74 * cm, f"{row['landmark']} {row['name']}", fontName="ReportFont-Bold", fontSize=7.4, fillColor=colors.black))
        drawing.add(String(x + 0.15 * cm, y + 0.42 * cm, f"mean {fmt(row['mean'])}", fontName="ReportFont", fontSize=7.0, fillColor=colors.black))
        drawing.add(String(x + 0.15 * cm, y + 0.16 * cm, f"med {fmt(row['median'])}", fontName="ReportFont", fontSize=7.0, fillColor=colors.black))
    legend_y = 0.05 * cm
    legend_x = 0.05 * cm
    steps = 60
    legend_w = 6.4 * cm
    for i in range(steps):
        value = low + (high - low) * i / (steps - 1)
        drawing.add(Rect(legend_x + legend_w * i / steps, legend_y, legend_w / steps + 0.5, 0.16 * cm, fillColor=color_interp(value, low, high), strokeColor=None))
    drawing.add(String(legend_x, legend_y + 0.24 * cm, f"Düşük {fmt(low)}", fontName="ReportFont", fontSize=6.8, fillColor=colors.HexColor("#4c5661")))
    drawing.add(String(legend_x + legend_w - 1.5 * cm, legend_y + 0.24 * cm, f"Yüksek {fmt(high)}", fontName="ReportFont", fontSize=6.8, fillColor=colors.HexColor("#4c5661")))
    return drawing


def group_table(group_stats):
    data = [["Anatomik bölge", "Landmarklar", "Mean", "Medyan", "Max"]]
    for row in group_stats:
        data.append([row["group"], row["landmarks"], fmt(row["mean"]), fmt(row["median"]), fmt(row["max"])])
    return styled_table(data, col_widths=[3.0 * cm, 7.0 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm], font_size=7.3)


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("ReportFont", 8)
    canvas.setFillColor(colors.HexColor("#6f7780"))
    canvas.drawString(1.8 * cm, 1.05 * cm, "PAL-Net ortodontik landmark sonuç raporu")
    canvas.drawRightString(19.2 * cm, 1.05 * cm, f"Sayfa {doc.page}")
    canvas.restoreState()


def build():
    register_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_json(RUN_RAW / "metrics.json")
    aligned_old = load_json(RUN_ALIGNED_OLD / "metrics.json")
    short = load_json(RUN_SHORT / "metrics.json")
    latest = load_json(RUN_LATEST / "metrics.json")
    transform = load_json(TRANSFORM_LATEST / "transform_report.json")
    group_rows = load_group_metrics(RUN_LATEST / "group_metrics_test.csv")
    sample_rows = load_sample_errors(RUN_LATEST / "predictions_test.csv")
    landmark_rows = load_landmark_stats(RUN_LATEST / "predictions_test.csv")
    group_stats = anatomical_group_stats(landmark_rows)
    history = load_json(RUN_LATEST / "history.json")
    epoch_count = len(history)
    best_epoch = min(history, key=lambda row: row["val_loss"])

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleTR",
            parent=styles["Title"],
            fontName="ReportFont-Bold",
            fontSize=22,
            leading=27,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17212b"),
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="HeadingTR",
            parent=styles["Heading1"],
            fontName="ReportFont-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#17212b"),
            spaceBefore=12,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyTR",
            parent=styles["BodyText"],
            fontName="ReportFont",
            fontSize=9.7,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallTR",
            parent=styles["BodyText"],
            fontName="ReportFont",
            fontSize=8.3,
            leading=11.5,
            textColor=colors.HexColor("#4c5661"),
            spaceAfter=6,
        )
    )

    story = []
    story.append(paragraph("PAL-Net ile Ortodontik 3B Landmark Lokalizasyonu", styles["TitleTR"]))
    story.append(paragraph("Sonuç Raporu", styles["HeadingTR"]))
    story.append(
        paragraph(
            "Bu rapor, 23 yumuşak doku landmarkı içeren 3B ortodontik yüz veri seti üzerinde PAL-Net modelinin son eğitim sonuçlarını özetler. "
            "Değerlendirme metriği Average Localization Error (ALE) olarak alınmıştır; ALE, PAL-Net tahmini ile uzman ortodontist işaretlemesi arasındaki 3B Öklid mesafesinin ortalamasıdır.",
            styles["BodyTR"],
        )
    )
    story.append(
        paragraph(
            "Son koşuda veri seti Trimesh tabanlı Procrustes hizalama matrisleriyle PAL-Net formatına yaklaştırılmıştır. "
            f"Nihai model paper'a daha yakın eğitim ayarlarıyla çalıştırılmıştır: patch_size=1000, surface_points=100000 ve {epoch_count} epoch.",
            styles["BodyTR"],
        )
    )

    overview = [
        ["Öğe", "Değer"],
        ["Son run", "20260627_143801 - patch1000 / surface100k"],
        ["Train / validation / test", f"{latest['n_train']} / {latest['n_val']} / {latest['n_test']}"],
        ["Landmark sayısı", "23"],
        ["Eğitim ayarı", f"patch_size=1000, surface_points=100000, epoch={epoch_count}"],
        ["En iyi validation epoch", f"{best_epoch['epoch']} - val_loss {fmt(best_epoch['val_loss'])}"],
        ["Ana metrik", "Average Localization Error - 3B Öklid mesafesi"],
        ["PAL-Net snapped ALE", fmt(latest["palnet_snapped"]["ale"])],
        ["PAL-Net snapped medyan", fmt(latest["palnet_snapped"]["median"])],
    ]
    story.append(styled_table(overview, col_widths=[5.2 * cm, 10.2 * cm]))

    story.append(paragraph("Yöntem ve Ön İşleme", styles["HeadingTR"]))
    story.append(
        paragraph(
            "PAL-Net kaynak kodundaki LA-FAS veri yükleyicisi, her mesh ve landmark dosyasına transformation_matrix.npy uygular. "
            "Bu nedenle ham PLY dosyalarını doğrudan modele vermek yeterli değildir; yüzlerin ortak bir koordinat sistemine taşınması gerekir. "
            "Bu çalışma için her örneğin 23 uzman landmarkı, Trimesh registration.procrustes fonksiyonu ile mean-template landmark şablonuna rigid olarak hizalanmıştır.",
            styles["BodyTR"],
        )
    )
    story.append(
        paragraph(
            "Rigid Procrustes dönüşümünde ölçekleme ve yansıma kapalı tutulmuştur. Böylece yüzlerin anatomik yönelimi korunur ve ALE değerleri veri setinin orijinal uzunluk birimiyle tutarlı kalır. "
            "Üretilen 4x4 matrisler hem mesh noktalarına hem de landmark koordinatlarına aynı şekilde uygulanmıştır.",
            styles["BodyTR"],
        )
    )
    story.append(
        styled_table(
            [
                ["Transform kalite ölçümü", "Ortalama", "Medyan", "P95", "Maksimum"],
                [
                    "Template hatası - önce",
                    fmt(transform["template_error_before"]["mean"]),
                    fmt(transform["template_error_before"]["median"]),
                    fmt(transform["template_error_before"]["p95"]),
                    fmt(transform["template_error_before"]["max"]),
                ],
                [
                    "Template hatası - sonra",
                    fmt(transform["template_error_after"]["mean"]),
                    fmt(transform["template_error_after"]["median"]),
                    fmt(transform["template_error_after"]["p95"]),
                    fmt(transform["template_error_after"]["max"]),
                ],
                [
                    "Landmark-yüzey mesafesi",
                    fmt(transform["landmark_to_transformed_surface"]["mean"]),
                    fmt(transform["landmark_to_transformed_surface"]["median"]),
                    fmt(transform["landmark_to_transformed_surface"]["p95"]),
                    fmt(transform["landmark_to_transformed_surface"]["max"]),
                ],
            ],
            col_widths=[6.0 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm],
        )
    )

    story.append(PageBreak())
    story.append(paragraph("Genel Sonuçlar", styles["HeadingTR"]))
    story.append(
        paragraph(
            f"Son uzun eğitimde PAL-Net snapped ALE değeri {fmt(latest['palnet_snapped']['ale'])} olarak ölçülmüştür. "
            "Snapped sonuç, ağın ürettiği landmarkın en yakın örneklenmiş yüzey noktasına taşınmış halidir. "
            "Bu değer, ham ve hizalanmamış koşudaki 13.304 ALE seviyesine göre çok büyük bir iyileşme göstermektedir.",
            styles["BodyTR"],
        )
    )
    story.append(metric_cards(latest))
    story.append(Spacer(1, 0.25 * cm))
    story.append(comparison_chart(raw, aligned_old, short, latest))
    story.append(Spacer(1, 0.25 * cm))
    story.append(
        styled_table(
            [
                ["Koşu", "Snapped ALE", "Medyan", "Kısa yorum"],
                ["Ham PLY, transform yok", fmt(raw["palnet_snapped"]["ale"]), fmt(raw["palnet_snapped"]["median"]), "Yüzler ortak koordinatta değil"],
                ["Procrustes transform - önceki", fmt(aligned_old["palnet_snapped"]["ale"]), fmt(aligned_old["palnet_snapped"]["median"]), "Paper bandına yaklaştı"],
                ["Procrustes + kısa eğitim", fmt(short["palnet_snapped"]["ale"]), fmt(short["palnet_snapped"]["median"]), "patch100 / surface5000"],
                ["Procrustes + uzun eğitim", fmt(latest["palnet_snapped"]["ale"]), fmt(latest["palnet_snapped"]["median"]), "patch1000 / surface100k"],
            ],
            col_widths=[4.7 * cm, 3.0 * cm, 2.2 * cm, 5.9 * cm],
            font_size=8.1,
        )
    )
    story.append(
        paragraph(
            "Bu karşılaştırma, performans kaybının PLY formatından değil, hizalama bilgisinin eksikliğinden kaynaklandığını destekler. "
            "Trimesh ile üretilen transformation_matrix.npy yapısı ve paper'a daha yakın eğitim yoğunluğu eklendiğinde sonuçlar 3.5 mm bandının altına inmiştir.",
            styles["BodyTR"],
        )
    )

    story.append(paragraph("Sınıf ve Cinsiyet Bazında Dağılım", styles["HeadingTR"]))
    bars = []
    for row in group_rows:
        bars.append(
            {
                "group": f"{row['class']} {row['gender']}",
                "ale": row["ale"],
                "std": row["std"],
                "median": row["median"],
            }
    )
    best_group = min(bars, key=lambda row: float(row["ale"]))
    worst_group = max(bars, key=lambda row: float(row["ale"]))
    story.append(horizontal_bar_chart(bars, "group", "ale", "Sınıf ve Cinsiyet Bazında ALE", max_value=max(float(r["ale"]) for r in bars)))
    story.append(
        paragraph(
            f"Grup kırılımında en düşük ALE {best_group['group']} grubunda {fmt(best_group['ale'])}, "
            f"en yüksek ALE ise {worst_group['group']} grubunda {fmt(worst_group['ale'])} olarak görülmektedir. "
            "Grupların birbirine yakın bir bantta kalması, uzun eğitim sonrası model davranışının sınıflar arasında görece dengeli olduğunu gösterir.",
            styles["BodyTR"],
        )
    )

    story.append(PageBreak())
    story.append(paragraph("Anatomik Bölgelere Göre Hata", styles["HeadingTR"]))
    story.append(
        paragraph(
            "Landmarklar anatomik anlamlarına göre beş bölgeye ayrılmıştır: orta hat profil, burun/filtrum, dudak-ağız, göz çevresi ve mandibular açı. "
            "Bu gruplama, tek tek noktaların ötesinde hangi yüz bölgelerinin model için daha kolay veya daha zor olduğunu okumayı sağlar.",
            styles["BodyTR"],
        )
    )
    story.append(horizontal_bar_chart(group_stats, "group", "mean", "Anatomik Bölge Mean ALE", max_value=max(row["mean"] for row in group_stats)))
    story.append(Spacer(1, 0.25 * cm))
    story.append(group_table(group_stats))
    easiest_group = min(group_stats, key=lambda row: row["mean"])
    hardest_group = max(group_stats, key=lambda row: row["mean"])
    story.append(
        paragraph(
            f"En düşük bölgesel hata {easiest_group['group']} grubunda {fmt(easiest_group['mean'])} olarak ölçülmüştür. "
            f"En yüksek bölgesel hata ise {hardest_group['group']} grubundadır ({fmt(hardest_group['mean'])}). "
            "Mandibular açı bölgesinin daha zor olması, Go_R ve Go_L noktalarındaki yüksek nokta bazlı hatayla uyumludur.",
            styles["BodyTR"],
        )
    )

    story.append(PageBreak())
    story.append(paragraph("Landmark Bazlı Hata Analizi", styles["HeadingTR"]))
    story.append(
        paragraph(
            "Aşağıdaki tabloda her bir landmark için test seti üzerindeki ortalama, medyan, standart sapma ve maksimum lokalizasyon hatası verilmiştir. "
            "Ortalama hata modelin genel sapmasını, medyan hata tipik vaka performansını, maksimum hata ise uç örnekleri gösterir.",
            styles["BodyTR"],
        )
    )
    story.append(landmark_heatmap(landmark_rows))
    best_lm = min(landmark_rows, key=lambda row: row["mean"])
    worst_lm = max(landmark_rows, key=lambda row: row["mean"])
    story.append(
        paragraph(
            f"Isı haritasında yeşil tonlar düşük, kırmızı tonlar yüksek ortalama hatayı gösterir. "
            f"En kolay nokta {best_lm['landmark']} ({best_lm['name']}), en zor nokta ise {worst_lm['landmark']} ({worst_lm['name']}) olarak öne çıkmaktadır.",
            styles["BodyTR"],
        )
    )
    story.append(PageBreak())
    story.append(paragraph("Landmark Detay Tablosu", styles["HeadingTR"]))
    story.append(Spacer(1, 0.25 * cm))
    lm_table_1 = [["LM", "Ad", "Mean", "Medyan", "Std", "Max"]]
    lm_table_2 = [["LM", "Ad", "Mean", "Medyan", "Std", "Max"]]
    for row in landmark_rows[:12]:
        lm_table_1.append([row["landmark"], row["name"], fmt(row["mean"]), fmt(row["median"]), fmt(row["std"]), fmt(row["max"])])
    for row in landmark_rows[12:]:
        lm_table_2.append([row["landmark"], row["name"], fmt(row["mean"]), fmt(row["median"]), fmt(row["std"]), fmt(row["max"])])
    story.append(styled_table(lm_table_1, col_widths=[1.2 * cm, 2.3 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm], font_size=8.2))
    story.append(Spacer(1, 0.25 * cm))
    story.append(styled_table(lm_table_2, col_widths=[1.2 * cm, 2.3 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm], font_size=8.2))
    story.append(
        paragraph(
            f"En düşük ortalama hata landmark {best_lm['landmark']} ({best_lm['name']}) için {fmt(best_lm['mean'])} olarak ölçülmüştür. "
            f"En yüksek ortalama hata ise landmark {worst_lm['landmark']} ({worst_lm['name']}) için {fmt(worst_lm['mean'])} değerindedir. "
            "Özellikle mandibular açı bölgesindeki Go_R ve Go_L noktaları diğer landmarklara göre daha zor görünmektedir.",
            styles["BodyTR"],
        )
    )

    story.append(PageBreak())
    story.append(paragraph("Örnek Bazında En İyi ve En Zor Vakalar", styles["HeadingTR"]))
    sample_table = [["Kategori", "Sample", "ALE", "Maksimum hata"]]
    for label, row in [
        ("En iyi", sample_rows[0]),
        ("İyi", sample_rows[1]),
        ("Medyan civarı", sample_rows[len(sample_rows) // 2]),
        ("Zor", sample_rows[-2]),
        ("En zor", sample_rows[-1]),
    ]:
        sample_table.append([label, row["sample_id"], fmt(row["ale"]), fmt(row["max"])])
    story.append(styled_table(sample_table, col_widths=[3.2 * cm, 4.2 * cm, 3.0 * cm, 3.2 * cm]))
    story.append(
        paragraph(
            "3B HTML görselleri, uzman işaretleri ile PAL-Net tahminleri arasındaki farkı yüz point-cloud'u üzerinde incelemek için üretilmiştir. "
            "Son uzun koşuya ait en iyi, medyan ve en zor örnek görselleri ilgili run klasörünün visualizations/index.html dosyasındadır.",
            styles["SmallTR"],
        )
    )

    story.append(PageBreak())
    story.append(paragraph("Yorum ve Sınırlılıklar", styles["HeadingTR"]))
    story.append(
        paragraph(
            "Sonuçların en önemli yorumu şudur: PAL-Net performansı, veri formatından çok hizalama kalitesine duyarlıdır. "
            f"Ham PLY dosyalarıyla ALE 13 seviyesindeyken, Trimesh Procrustes transformation_matrix.npy yapısı ve paper'a yakın eğitim ayarları eklendiğinde ALE {fmt(latest['palnet_snapped']['ale'])} seviyesine inmiştir. "
            "Bu, PAL-Net'in patch tabanlı yapısının doğru başlangıç landmark merkezlerine ve yeterli patch yoğunluğuna ihtiyaç duyduğunu gösterir.",
            styles["BodyTR"],
        )
    )
    story.append(
        paragraph(
            "Bununla birlikte bu rapordaki Procrustes transform, uzman landmarklarını kullanarak üretilmiştir. "
            "Dolayısıyla sonuç, klinikte tamamen bağımsız bir landmark tespit performansı olarak değil, landmark tabanlı hizalama sonrası PAL-Net'in uzman işaretlerine ne kadar yaklaştığını gösteren kontrollü bir deney olarak yorumlanmalıdır.",
            styles["BodyTR"],
        )
    )
    story.append(
        paragraph(
            "Daha katı bir klinik değerlendirme için test örneklerinde uzman landmarklarını hizalama aşamasında kullanmayan bir registration pipeline önerilir. "
            "Örneğin mesh tabanlı ICP, burun/orta yüz gibi anatomik sabit bölgelerle rigid hizalama veya eğitim setinden öğrenilen şablon bazlı otomatik hizalama kullanılabilir. "
            "Bu durumda PAL-Net'in gerçek genelleme performansı daha dürüst biçimde ölçülebilir.",
            styles["BodyTR"],
        )
    )
    story.append(
        paragraph(
            "Bir sonraki teknik adım, test örneklerinde uzman landmarklarını kullanmadan otomatik registration üretmek ve aynı uzun eğitim protokolünü bu daha katı senaryoda tekrar etmektir. "
            "Ayrıca Go_R ve Go_L gibi yüksek hatalı landmarklar için nokta özelinde veri kalitesi ve anatomik tutarlılık kontrolü yapılmalıdır.",
            styles["BodyTR"],
        )
    )

    doc = SimpleDocTemplate(
        str(OUT_PATH),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="PAL-Net Ortodontik Sonuç Raporu",
        author="Codex",
    )
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    print(OUT_PATH)


if __name__ == "__main__":
    build()
