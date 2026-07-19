import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "pdf"
ASSET_DIR = OUT_DIR / "three_model_report_assets"
PDF_PATH = OUT_DIR / "uc_model_performans_karsilastirma_raporu.pdf"

RUNS = {
    "PAL-Net": {
        "path": ROOT / "palnet_orthodontic_comparison" / "runs" / "palnet_cpu_prediction_stage1_patch1000_surface100k",
        "metrics": "metrics.json",
        "metric_key": "palnet_snapped",
        "short": "Patch Attention Landmark Network",
        "run_label": "palnet_cpu_prediction_stage1_patch1000_surface100k",
    },
    "DiffusionNet": {
        "path": ROOT
        / "diffusion_net_orthodontic_comparison"
        / "runs"
        / "diffusionnet_shared_metrics_p12000_k96_w192_b8_e220_topk30",
        "metrics": "metrics.json",
        "metric_key": "diffusionnet_heatmap",
        "short": "DiffusionNet heatmap segmentation",
        "run_label": "diffusionnet_shared_metrics_p12000_k96_w192_b8_e220_topk30",
    },
    "PointNet++": {
        "path": ROOT / "pointnet2_orthodontic_comparison" / "runs" / "pointnet2_shared_metrics_p4096_e200_topk20",
        "metrics": "metrics.json",
        "metric_key": "pointnet2",
        "short": "PointNet++ heatmap segmentation",
        "run_label": "pointnet2_shared_metrics_p4096_e200_topk20",
    },
}

MODEL_DESCRIPTIONS = {
    "PAL-Net": (
        "PAL-Net, 3B yuz uzerinde her landmark icin lokal patch cikaran ve patch icindeki noktalari "
        "attention mekanizmasi ile agirliklandiran nokta tabanli bir CNN yaklasimidir. Bu raporda PAL-Net "
        "ciktisi yuzey noktasina snap edilmis tahminler uzerinden degerlendirildi."
    ),
    "DiffusionNet": (
        "DiffusionNet, mesh veya nokta bulutu uzerinde geometrik diffusion operatorlerini kullanarak "
        "yerel ve global sekil bilgisini isleyen bir mimaridir. Bu calismada landmark lokalizasyonu "
        "23 kanalli heatmap/mask tahmini ve top-k agirlikli postprocess olarak ele alindi."
    ),
    "PointNet++": (
        "PointNet++, nokta bulutunu hiyerarsik set abstraction ve feature propagation bloklariyla isler. "
        "Bu raporda yuzeyden orneklenen noktalar ve normaller kullanildi; her landmark icin gaussian heatmap "
        "tahmin edilip top-k softmax ile koordinata donusturuldu."
    ),
}


def register_fonts():
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    bold_candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    regular = next((p for p in candidates if Path(p).exists()), None)
    bold = next((p for p in bold_candidates if Path(p).exists()), regular)
    if regular:
        pdfmetrics.registerFont(TTFont("ReportFont", regular))
        pdfmetrics.registerFont(TTFont("ReportFont-Bold", bold))
        return "ReportFont", "ReportFont-Bold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = register_fonts()


def load_data():
    data = {}
    for model, cfg in RUNS.items():
        metrics_path = cfg["path"] / cfg["metrics"]
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary = metrics[cfg["metric_key"]]
        data[model] = {
            "cfg": cfg,
            "metrics": metrics,
            "summary": summary,
        }
    return data


def fmt(value, digits=3):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def pct(value):
    return f"{float(value) * 100:.1f}%"


def make_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName=FONT_BOLD,
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17324D"),
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.5,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#475569"),
            spaceAfter=20,
        ),
        "h1": ParagraphStyle(
            "Heading1",
            parent=base["Heading1"],
            fontName=FONT_BOLD,
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#17324D"),
            spaceBefore=10,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "Heading2",
            parent=base["Heading2"],
            fontName=FONT_BOLD,
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#1F4E5F"),
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.4,
            leading=13.5,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#111827"),
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=7.8,
            leading=10,
            textColor=colors.HexColor("#475569"),
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=8.0,
            leading=10.0,
            textColor=colors.HexColor("#111827"),
        ),
        "table_head": ParagraphStyle(
            "TableHead",
            parent=base["BodyText"],
            fontName=FONT_BOLD,
            fontSize=7.8,
            leading=9.5,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
    }


def p(text, styles, style="body"):
    return Paragraph(text, styles[style])


def table(data, widths=None, header=True, font_size=7.4):
    rows = []
    for ridx, row in enumerate(data):
        cells = []
        for cell in row:
            style_name = "table_head" if ridx == 0 and header else "table_cell"
            cells.append(Paragraph(str(cell), STYLES[style_name]))
        rows.append(cells)
    tbl = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
            ]
        )
        for r in range(1, len(rows)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#F8FAFC")))
    tbl.setStyle(TableStyle(style))
    return tbl


def save_chart(path, fig):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def create_charts(data):
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    names = list(data)
    colors_ = ["#2F6B7C", "#8A5A44", "#4C7A3F"]

    ale = [data[n]["summary"]["ale"] for n in names]
    med = [data[n]["summary"]["median"] for n in names]
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )

    fig, ax = plt.subplots(figsize=(4.8, 2.25))
    x = np.arange(len(names))
    width = 0.36
    ax.bar(x - width / 2, ale, width, label="ALE", color="#2F6B7C")
    ax.bar(x + width / 2, med, width, label="Median", color="#8A5A44")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Hata (mm)")
    ax.set_title("Genel lokalizasyon hatasi")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for i, v in enumerate(ale):
        ax.text(i - width / 2, v + 0.04, fmt(v, 2), ha="center", fontsize=8)
    for i, v in enumerate(med):
        ax.text(i + width / 2, v + 0.04, fmt(v, 2), ha="center", fontsize=8)
    overview = save_chart(ASSET_DIR / "overall_error.png", fig)

    fig, ax = plt.subplots(figsize=(4.8, 2.25))
    pcks = ["pck_at_2mm", "pck_at_2_5mm", "pck_at_3mm"]
    labels = ["PCK@2", "PCK@2.5", "PCK@3"]
    x = np.arange(len(labels))
    width = 0.24
    for idx, name in enumerate(names):
        vals = [data[name]["summary"][key] * 100 for key in pcks]
        ax.bar(x + (idx - 1) * width, vals, width, label=name, color=colors_[idx])
        for j, v in enumerate(vals):
            ax.text(x[j] + (idx - 1) * width, v + 1.2, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 82)
    ax.set_ylabel("Basari orani (%)")
    ax.set_title("Klinik esik performansi")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    pck = save_chart(ASSET_DIR / "pck_thresholds.png", fig)

    fig, ax = plt.subplots(figsize=(4.8, 2.25))
    class_labels = ["Class I", "Class II", "Class III"]
    x = np.arange(len(class_labels))
    width = 0.24
    for idx, name in enumerate(names):
        rows = {r["class"]: r for r in data[name]["metrics"]["class_performance"]}
        vals = [rows[label]["mean"] for label in class_labels]
        ax.bar(x + (idx - 1) * width, vals, width, label=name, color=colors_[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels)
    ax.set_ylabel("ALE (mm)")
    ax.set_title("Angle siniflarina gore performans")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    class_chart = save_chart(ASSET_DIR / "class_performance.png", fig)

    fig, ax = plt.subplots(figsize=(5.2, 2.35))
    lm = np.arange(23)
    for idx, name in enumerate(names):
        vals = data[name]["summary"]["per_landmark_ale"]
        ax.plot(lm, vals, marker="o", linewidth=1.8, markersize=3.8, label=name, color=colors_[idx])
    ax.set_xticks(lm)
    ax.set_xlabel("Landmark indeksi")
    ax.set_ylabel("Mean hata (mm)")
    ax.set_title("Landmark bazli hata profili")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    landmark = save_chart(ASSET_DIR / "landmark_profile.png", fig)

    fig, ax = plt.subplots(figsize=(5.2, 1.8))
    matrix = np.asarray([data[name]["summary"]["per_landmark_ale"] for name in names])
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xticks(range(23))
    ax.set_xticklabels([str(i) for i in range(23)], fontsize=7)
    ax.set_xlabel("Landmark")
    ax.set_title("Landmark hata isi haritasi (mm)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.set_ylabel("mm", rotation=270, labelpad=10)
    heatmap = save_chart(ASSET_DIR / "landmark_heatmap.png", fig)

    fig, ax = plt.subplots(figsize=(4.8, 2.25))
    top_landmarks = [21, 22, 0, 16, 12]
    x = np.arange(len(top_landmarks))
    width = 0.24
    for idx, name in enumerate(names):
        vals = [data[name]["summary"]["per_landmark_ale"][lm_idx] for lm_idx in top_landmarks]
        ax.bar(x + (idx - 1) * width, vals, width, label=name, color=colors_[idx])
    ax.set_xticks(x)
    ax.set_xticklabels([f"LM{i}" for i in top_landmarks])
    ax.set_ylabel("Mean hata (mm)")
    ax.set_title("Ortak zor landmarklar")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=7)
    hard_landmarks = save_chart(ASSET_DIR / "hard_landmarks.png", fig)

    return {
        "overview": overview,
        "pck": pck,
        "class": class_chart,
        "landmark": landmark,
        "heatmap": heatmap,
        "hard_landmarks": hard_landmarks,
    }


def add_image(story, path, width_cm=10.5, max_height_cm=5.2):
    img = Image(str(path))
    target_width = width_cm * cm
    target_height = img.imageHeight * target_width / img.imageWidth
    max_height = max_height_cm * cm
    if target_height > max_height:
        target_height = max_height
        target_width = img.imageWidth * target_height / img.imageHeight
    img.drawWidth = target_width
    img.drawHeight = target_height
    story.append(img)
    story.append(Spacer(1, 0.2 * cm))


def chart_block(story, title, note, path, width_cm=10.5, max_height_cm=5.2):
    story.append(KeepTogether([p(title, STYLES, "h2"), p(note, STYLES, "small")]))
    add_image(story, path, width_cm, max_height_cm)


def build_story(data, charts):
    story = []
    story.append(p("3B Yuz Landmark Lokalizasyonu: PAL-Net, DiffusionNet ve PointNet++ Karsilastirmasi", STYLES, "title"))
    story.append(
        p(
            "Ortak 180 egitim / 60 validasyon / 60 test bolunmesiyle 23 uzman landmark uzerinde "
            "Average Localization Error, PCK esikleri, sinif-cinsiyet kirilimlari ve zor landmark analizi.",
            STYLES,
            "subtitle",
        )
    )

    story.append(p("1. Calismanin Kapsami", STYLES, "h1"))
    story.append(
        p(
            "Bu dokuman, ayni ortodontik 3B yuz veri seti uzerinde calistirilan uc acik kaynak model "
            "ailesinin guncel performansini ozetler. Degerlendirme test setindeki 60 hasta ve her hasta icin "
            "23 yumuşak doku landmarki uzerinden yapildi. Ana metrik, uzman isaretlemesi ile model tahmini "
            "arasindaki Oklid uzakliginin ortalamasidir (ALE, mm).",
            STYLES,
        )
    )
    story.append(
        p(
            "Klinik esik yorumu icin PCK@2mm, PCK@2.5mm ve PCK@3mm raporlandi. Ayrica Angle Class I/II/III, "
            "cinsiyet, class-gender ve landmark bazli ayrintilar ayni shared metrics formatindan uretildi.",
            STYLES,
        )
    )

    story.append(p("2. Model Ozeti", STYLES, "h1"))
    model_rows = [["Model", "Kisa tanim", "Kullanilan run"]]
    for model, item in data.items():
        model_rows.append([model, MODEL_DESCRIPTIONS[model], item["cfg"]["run_label"]])
    story.append(table(model_rows, widths=[2.4 * cm, 9.2 * cm, 5.2 * cm]))

    story.append(p("3. Deney Protokolu", STYLES, "h1"))
    protocol_rows = [["Alan", "Deger"]]
    first = next(iter(data.values()))["metrics"]
    protocol_rows.extend(
        [
            ["Veri bolunmesi", f"{first.get('n_train')} egitim / {first.get('n_val')} validasyon / {first.get('n_test')} test"],
            ["Landmark sayisi", "23"],
            ["Ana metrik", "ALE - 23 landmark uzerindeki ortalama Oklid uzakligi"],
            ["Klinik esikler", "PCK@2mm, PCK@2.5mm, PCK@3mm"],
            ["Hizalama", "Procrustes rigid transform klasoru kullanilan aligned protokol"],
        ]
    )
    story.append(table(protocol_rows, widths=[4.2 * cm, 12.6 * cm]))

    story.append(p("4. Ana Performans Sonuclari", STYLES, "h1"))
    main_rows = [["Model", "ALE", "Median", "Std", "Max", "PCK@2", "PCK@2.5", "PCK@3"]]
    for model, item in sorted(data.items(), key=lambda kv: kv[1]["summary"]["ale"]):
        s = item["summary"]
        main_rows.append(
            [
                model,
                fmt(s["ale"]),
                fmt(s["median"]),
                fmt(s["std"]),
                fmt(s["max"]),
                pct(s["pck_at_2mm"]),
                pct(s["pck_at_2_5mm"]),
                pct(s["pck_at_3mm"]),
            ]
        )
    story.append(table(main_rows, widths=[2.9 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm, 2 * cm]))
    story.append(
        p(
            "ALE acisindan PAL-Net en dusuk ortalama hatayi verirken, PointNet++ median hata ve 2.5/3 mm "
            "esik basarisinda daha guclu gorunmektedir. DiffusionNet bu kosuda ozellikle zor landmarklardaki "
            "outlier etkisi nedeniyle daha yuksek ALE gostermistir.",
            STYLES,
        )
    )
    story.append(PageBreak())
    story.append(p("5. Genel Grafikler", STYLES, "h1"))
    chart_block(
        story,
        "ALE ve median hata",
        "PAL-Net ortalama ALE'de, PointNet++ median hatada one cikmaktadir.",
        charts["overview"],
        10.2,
        4.8,
    )
    chart_block(
        story,
        "Klinik esik basarisi",
        "PCK degeri, tahmin edilen landmarklarin ilgili mm esigi icinde kalma oranidir.",
        charts["pck"],
        10.2,
        4.8,
    )

    story.append(PageBreak())
    story.append(p("6. Model Konfigurasyonlari", STYLES, "h1"))
    cfg_rows = [["Model", "Temel ayarlar"]]
    pal = data["PAL-Net"]["metrics"]
    diff = data["DiffusionNet"]["metrics"]
    pn = data["PointNet++"]["metrics"]
    cfg_rows.extend(
        [
            [
                "PAL-Net",
                "patch_size=1000, surface_points=100000, stage1 checkpoint, snapped surface prediction, template_mode=global",
            ],
            [
                "DiffusionNet",
                f"surface_points={diff.get('surface_points')}, k_eig={diff.get('k_eig')}, width={diff.get('width')}, "
                f"blocks={diff.get('blocks')}, loss={diff.get('loss_mode')}, postprocess={diff.get('postprocess')}, topk={diff.get('refine_topk')}",
            ],
            [
                "PointNet++",
                f"surface_points={pn.get('surface_points')}, eval_points={pn.get('eval_surface_points')}, "
                f"SA=({pn.get('sa1_points')},{pn.get('sa2_points')},{pn.get('sa3_points')}), normals={pn.get('use_normals')}, "
                f"postprocess={pn.get('postprocess')}, topk={pn.get('topk')}",
            ],
        ]
    )
    story.append(table(cfg_rows, widths=[3 * cm, 13.8 * cm]))

    story.append(p("7. Sinif Bazli Performans", STYLES, "h1"))
    class_rows = [["Model", "Class I ALE", "Class II ALE", "Class III ALE", "En iyi sinif", "En zor sinif"]]
    for model, item in data.items():
        rows = {r["class"]: r for r in item["metrics"]["class_performance"]}
        vals = {label: rows[label]["mean"] for label in ["Class I", "Class II", "Class III"]}
        best = min(vals, key=vals.get)
        worst = max(vals, key=vals.get)
        class_rows.append([model, fmt(vals["Class I"]), fmt(vals["Class II"]), fmt(vals["Class III"]), best, worst])
    story.append(table(class_rows, widths=[3 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 3.2 * cm, 3.2 * cm]))
    chart_block(
        story,
        "Angle siniflarina gore ALE",
        "Siniflar arasi farklar modelden modele degisse de buyuk performans kopmasi yoktur.",
        charts["class"],
        10.2,
        4.8,
    )

    story.append(p("8. Cinsiyet Bazli Performans", STYLES, "h1"))
    gender_rows = [["Model", "Female ALE", "Female PCK@2", "Male ALE", "Male PCK@2", "Yorum"]]
    for model, item in data.items():
        rows = {r["gender"]: r for r in item["metrics"]["gender_performance"]}
        female = rows["female"]
        male = rows["male"]
        comment = "Female alt grup daha dusuk hata" if female["mean"] < male["mean"] else "Male alt grup daha dusuk hata"
        gender_rows.append([model, fmt(female["mean"]), pct(female["pck_at_2mm"]), fmt(male["mean"]), pct(male["pck_at_2mm"]), comment])
    story.append(table(gender_rows, widths=[2.8 * cm, 2.2 * cm, 2.4 * cm, 2.2 * cm, 2.4 * cm, 4.8 * cm]))

    story.append(PageBreak())
    story.append(p("9. Landmark Bazli Hata Analizi", STYLES, "h1"))
    story.append(
        p(
            "Uc modelde de en zor noktalar ayni bolgede yogunlasmaktadir: LM21, LM22 ve LM0. Bu durum model "
            "mimarisinden bagimsiz olarak veri/landmark tanimi veya ilgili anatomik bolgenin geometrik belirsizligi "
            "ile iliskili olabilir.",
            STYLES,
        )
    )
    chart_block(
        story,
        "Ortak zor landmarklar",
        "LM21, LM22 ve LM0 uc modelde de genel hatayi yukari ceken ana noktalardir.",
        charts["hard_landmarks"],
        10.2,
        4.8,
    )
    chart_block(
        story,
        "Landmark hata profili",
        "23 landmark boyunca mean hata egilimleri; ayni indekslerde tekrarlayan yukselmeler anatomik zorluk sinyali verir.",
        charts["landmark"],
        11.0,
        5.0,
    )

    story.append(PageBreak())
    story.append(p("10. Landmark Isi Haritasi ve Zor Noktalar", STYLES, "h1"))
    chart_block(
        story,
        "Landmark hata isi haritasi",
        "Daha koyu renk daha yuksek mean hata anlamina gelir. LM21-LM22 bolgesi tum modellerde belirgindir.",
        charts["heatmap"],
        11.0,
        4.2,
    )

    hard_rows = [["Model", "1. zor", "Mean", "Median", "Max", "2. zor", "3. zor"]]
    for model, item in data.items():
        hard = item["metrics"]["difficult_landmark_analysis"]["all_landmarks"]
        hard_rows.append(
            [
                model,
                f"LM{hard[0]['landmark']}",
                fmt(hard[0]["mean"]),
                fmt(hard[0]["median"]),
                fmt(hard[0]["max"]),
                f"LM{hard[1]['landmark']} ({fmt(hard[1]['mean'])})",
                f"LM{hard[2]['landmark']} ({fmt(hard[2]['mean'])})",
            ]
        )
    story.append(table(hard_rows, widths=[2.7 * cm, 2 * cm, 1.8 * cm, 1.8 * cm, 1.9 * cm, 3.4 * cm, 3.4 * cm]))

    story.append(p("11. Klinik Esik Analizi", STYLES, "h1"))
    clinical_rows = [["Model", "2 mm icinde", "2.5 mm icinde", "3 mm icinde", "2 mm disinda", "3 mm disinda"]]
    for model, item in data.items():
        o = item["metrics"]["overall_threshold_performance"]
        clinical_rows.append(
            [
                model,
                f"{o['n_within_2mm']} / {o['n_points']} ({pct(o['pck_at_2mm'])})",
                f"{o['n_within_2_5mm']} / {o['n_points']} ({pct(o['pck_at_2_5mm'])})",
                f"{o['n_within_3mm']} / {o['n_points']} ({pct(o['pck_at_3mm'])})",
                pct(o["fail_rate_gt_2mm"]),
                pct(o["fail_rate_gt_3mm"]),
            ]
        )
    story.append(table(clinical_rows, widths=[2.7 * cm, 3.3 * cm, 3.3 * cm, 3.3 * cm, 2.1 * cm, 2.1 * cm]))

    story.append(PageBreak())
    story.append(p("12. Sonuc ve Karsilastirma", STYLES, "h1"))
    story.append(
        p(
            "Genel ALE siralamasi PAL-Net (2.574 mm), PointNet++ (2.607 mm) ve DiffusionNet (2.875 mm) seklindedir. "
            "Bu farklar PAL-Net ile PointNet++ arasinda kucuktur; buna karsilik DiffusionNet guncel kosuda daha "
            "belirgin outlier etkisi tasimistir.",
            STYLES,
        )
    )
    story.append(
        p(
            "Median hata acisindan PointNet++ 2.077 mm ile en iyi degeri uretmistir. PCK@2mm'de PAL-Net ve "
            "PointNet++ esit gorunurken, PCK@2.5mm ve PCK@3mm esiklerinde PointNet++ daha yuksek basari saglamistir. "
            "Bu nedenle klinik esik bazli yorumda PointNet++ daha dengeli bir profil sunar.",
            STYLES,
        )
    )
    story.append(
        p(
            "Uc modelin ortak zayif noktasi LM21, LM22 ve LM0 landmarklaridir. Makale tartismasinda bu landmarklarin "
            "anatomik tanim belirsizligi, yuzey sampling yogunlugu ve uzman isaretlerinin yuzey vertexleri ile iliskisi "
            "ayri bir hata kaynagi olarak ele alinmalidir.",
            STYLES,
        )
    )
    conclusion_rows = [["Kriter", "En iyi model", "Not"]]
    conclusion_rows.extend(
        [
            ["En dusuk ALE", "PAL-Net", "2.574 mm ile en dusuk ortalama hata"],
            ["En dusuk median", "PointNet++", "2.077 mm ile tipik hata daha dusuk"],
            ["PCK@2mm", "PAL-Net / PointNet++", "Iki model de 47.6%"],
            ["PCK@2.5 ve PCK@3", "PointNet++", "62.2% ve 71.8% ile en yuksek esik basarisi"],
            ["Outlier kontrolu", "PointNet++", "Max hata 15.33 mm; PAL-Net 18.34 mm; DiffusionNet 27.44 mm"],
        ]
    )
    story.append(table(conclusion_rows, widths=[4 * cm, 4 * cm, 8.8 * cm]))
    story.append(
        p(
            "Son yorum: Bu veri seti ve guncel protokolde hicbir model genel mean ALE'yi 2 mm altina indirmemistir. "
            "Buna ragmen PAL-Net ve PointNet++ 2.6 mm bandinda yakin performans gostermis, PointNet++ ise klinik "
            "esiklerde daha yuksek kapsama uretmistir. Makalede ana karsilastirma hem ALE hem PCK esikleriyle birlikte "
            "sunulmalidir; sadece tek bir ortalama hata metriğine dayanmak model davranisini eksik yansitir.",
            STYLES,
        )
    )
    story.append(Spacer(1, 0.5 * cm))
    story.append(
        p(
            "Kaynak dosyalar: PAL-Net metrics.json, DiffusionNet metrics.json, PointNet++ metrics.json ve ilgili "
            "landmark/class/gender/clinical threshold CSV ciktilari.",
            STYLES,
            "small",
        )
    )
    return story


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.white)
    canvas.rect(0, 0, A4[0], A4[1], stroke=0, fill=1)
    canvas.setFont(FONT, 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(1.6 * cm, 1.0 * cm, "3B yuz landmark model karsilastirma raporu")
    canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"Sayfa {doc.page}")
    canvas.restoreState()


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STYLES = make_styles()
    data = load_data()
    charts = create_charts(data)
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.45 * cm,
        leftMargin=1.45 * cm,
        topMargin=1.3 * cm,
        bottomMargin=1.45 * cm,
        title="Uc model performans karsilastirma raporu",
        author="comparative-study",
    )
    story = build_story(data, charts)
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(PDF_PATH)
