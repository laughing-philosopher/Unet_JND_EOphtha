"""report_v2.py
==============
Two-phase multilingual PDF report for Aakhi retinal analysis.

Structure: Full English report first, then the selected language after a page break.
Languages: en, hi, mr, or, bn, te, ta, gu, sat

Fonts required in fonts/ (run download_fonts.py once):
  NotoSans-Regular.ttf
  NotoSansDevanagari-Regular.ttf  (Hindi, Marathi)
  NotoSansOdia-Regular.ttf        (Odia)
  NotoSansBengali-Regular.ttf     (Bengali)
  NotoSansTelugu-Regular.ttf      (Telugu)
  NotoSansTamil-Regular.ttf       (Tamil)
  NotoSansGujarati-Regular.ttf    (Gujarati)
  NotoSansOlChiki-Regular.ttf     (Santali)
"""

import io
import os
import json
import uuid
from datetime import datetime

import numpy as np
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether, PageBreak,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Path helpers ──────────────────────────────────────────────────────────── #
_BASE_DIR = (os.environ.get('AAKHI_BASE_PATH') or
             os.path.dirname(os.path.abspath(__file__)))
FONTS_DIR = os.path.join(_BASE_DIR, "fonts")

# ── Font registration ─────────────────────────────────────────────────────── #
FONT_FILES = {
    "NotoSans":      "NotoSans-Regular.ttf",
    "NotoSansHi":    "NotoSansDevanagari-Regular.ttf",   # Hindi
    "NotoSansMr":    "NotoSansDevanagari-Regular.ttf",   # Marathi (same script)
    "NotoSansOr":    "NotoSansOdia-Regular.ttf",
    "NotoSansBn":    "NotoSansBengali-Regular.ttf",
    "NotoSansTe":    "NotoSansTelugu-Regular.ttf",
    "NotoSansTa":    "NotoSansTamil-Regular.ttf",
    "NotoSansGu":    "NotoSansGujarati-Regular.ttf",
    "NotoSansSat":   "NotoSansOlChiki-Regular.ttf",
}

_fonts_registered = False


def _register_fonts():
    global _fonts_registered
    if _fonts_registered:
        return
    for name, fname in FONT_FILES.items():
        path = os.path.join(FONTS_DIR, fname)
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
            except Exception:
                pass
    _fonts_registered = True


_LANG_FONT = {
    "en":  "NotoSans",
    "hi":  "NotoSansHi",
    "mr":  "NotoSansMr",
    "or":  "NotoSansOr",
    "bn":  "NotoSansBn",
    "te":  "NotoSansTe",
    "ta":  "NotoSansTa",
    "gu":  "NotoSansGu",
    "sat": "NotoSansSat",
}

_I18N_CACHE: dict[str, dict] = {}


def _t(lang: str, key: str, fallback: str = "") -> str:
    """Return translated string for key in lang, fall back to English."""
    if lang not in _I18N_CACHE:
        i18n_dir = os.path.join(_BASE_DIR, "static", "i18n")
        for code in (lang, "en"):
            fpath = os.path.join(i18n_dir, f"{code}.json")
            if os.path.exists(fpath):
                with open(fpath, encoding="utf-8") as f:
                    _I18N_CACHE[code] = json.load(f)
            if code == lang and lang in _I18N_CACHE:
                break
    data = _I18N_CACHE.get(lang) or _I18N_CACHE.get("en", {})
    return data.get(key, fallback or key)


# ── Image helpers ─────────────────────────────────────────────────────────── #

def _np_to_rl(arr: np.ndarray, max_w_mm: float = 140, max_h_mm: float = 180) -> RLImage:
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
    pil = PILImage.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    w, h = pil.size
    if w == 0 or h == 0:
        return RLImage(buf, width=50 * mm, height=50 * mm)
    scale = min((max_w_mm * mm) / w, (max_h_mm * mm) / h)
    return RLImage(buf, width=w * scale, height=h * scale)


def _side_by_side(img_a, label_a: str, img_b, label_b: str,
                  styles, max_half_mm: float = 85) -> Table:
    def _cell(arr, label):
        if arr is None:
            return [Paragraph("(not available)", styles["small_gray"])]
        return [_np_to_rl(arr, max_w_mm=max_half_mm, max_h_mm=160),
                Paragraph(label, styles["img_label"])]

    data = [[_cell(img_a, label_a), _cell(img_b, label_b)]]
    t = Table(data, colWidths=[90 * mm, 90 * mm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


# ── Severity / description data ───────────────────────────────────────────── #

DR_DESCRIPTIONS = {
    "No DR":            "No signs of diabetic retinopathy detected. The retina appears normal with no microaneurysms, haemorrhages, or exudates.",
    "Mild DR":          "Mild non-proliferative DR. Presence of at least one microaneurysm. Regular monitoring recommended.",
    "Moderate DR":      "Moderate non-proliferative DR. Multiple microaneurysms, dot/blot haemorrhages, and possible hard exudates. Referral to ophthalmologist advised.",
    "Severe DR":        "Severe non-proliferative DR (pre-proliferative). More than 20 intraretinal haemorrhages in each quadrant, venous beading, or intraretinal microvascular abnormalities. Urgent referral required.",
    "Proliferative DR": "Proliferative DR. Neovascularisation present. Risk of vitreous haemorrhage and tractional retinal detachment. Immediate ophthalmology review and treatment required.",
}

CDR_INTERPRETATION = {
    "normal":    "CDR within normal limits (< 0.5). Low suspicion for glaucomatous optic neuropathy.",
    "borderline":"Borderline CDR (0.5 – 0.7). Careful monitoring and visual field testing recommended.",
    "abnormal":  "Elevated CDR (> 0.7). Highly suspicious for glaucomatous damage. Formal glaucoma evaluation required.",
}

GLAUCOMA_DESCRIPTIONS = {
    "No Glaucoma":       "No evidence of glaucomatous optic neuropathy.",
    "Glaucoma Suspect":  "Suspicious features for glaucoma. Full glaucoma workup including IOP, visual fields, and OCT RNFL recommended.",
    "Moderate Glaucoma": "Moderate glaucomatous optic neuropathy with rim thinning. IOP management and regular follow-up essential.",
    "Advanced Glaucoma": "Advanced glaucomatous damage. Significant rim loss. Urgent IOP reduction and specialist management required.",
}

FOLLOW_UP = {
    0: "Annual screening recommended.",
    1: "Follow-up in 6–12 months. Optimise glycaemic and blood pressure control.",
    2: "Referral to ophthalmologist within 3–6 months.",
    3: "Urgent ophthalmology referral within 1–3 months.",
    4: "Immediate ophthalmology referral. Possible laser/surgical intervention.",
}


def _cdr_interpretation(vcdr: float) -> str:
    if vcdr < 0.5:
        key = "normal"
    elif vcdr <= 0.7:
        key = "borderline"
    else:
        key = "abnormal"
    return CDR_INTERPRETATION[key]


def _dr_level(grade_str: str) -> int:
    return {"No DR": 0, "Mild DR": 1, "Moderate DR": 2,
            "Severe DR": 3, "Proliferative DR": 4}.get(grade_str, 0)


# ── Style factory ─────────────────────────────────────────────────────────── #

def _make_styles(lang: str = "en") -> dict:
    _register_fonts()
    font = _LANG_FONT.get(lang, "NotoSans")
    try:
        pdfmetrics.getFont(font)
    except Exception:
        font = "NotoSans"
    try:
        pdfmetrics.getFont(font)
    except Exception:
        font = "Helvetica"

    ef = "NotoSans"   # English/Latin font — always used for technical English content
    try:
        pdfmetrics.getFont(ef)
    except Exception:
        ef = "Helvetica"

    return {
        # Native-script styles (section headers, translated labels, disclaimer)
        "title": ParagraphStyle("title",
            fontName=font, fontSize=18,
            textColor=colors.HexColor("#0d2b52"),
            alignment=TA_CENTER, spaceAfter=2 * mm),
        "subtitle": ParagraphStyle("subtitle",
            fontName=font, fontSize=10,
            textColor=colors.HexColor("#555"),
            alignment=TA_CENTER, spaceAfter=3 * mm),
        "section": ParagraphStyle("section",
            fontName=font, fontSize=12, leading=16,
            textColor=colors.HexColor("#0d2b52"),
            spaceBefore=5 * mm, spaceAfter=2 * mm),
        "body": ParagraphStyle("body",
            fontName=ef, fontSize=9, leading=13,
            spaceAfter=2 * mm),
        "small_gray": ParagraphStyle("small_gray",
            fontName=font, fontSize=8,
            textColor=colors.HexColor("#777"),
            alignment=TA_CENTER, spaceAfter=1 * mm),
        "img_label": ParagraphStyle("img_label",
            fontName=ef, fontSize=8,
            textColor=colors.HexColor("#555"),
            alignment=TA_CENTER, spaceAfter=2 * mm),
        "bold": ParagraphStyle("bold",
            fontName=ef, fontSize=9, leading=13,
            textColor=colors.HexColor("#222"),
            spaceAfter=1 * mm),
        "warn": ParagraphStyle("warn",
            fontName=ef, fontSize=9, leading=13,
            textColor=colors.HexColor("#b03a00"),
            spaceAfter=2 * mm),
        "ok": ParagraphStyle("ok",
            fontName=ef, fontSize=9, leading=13,
            textColor=colors.HexColor("#1a7a1a"),
            spaceAfter=2 * mm),
        "font":    font,   # native script font (for translated strings)
        "en_font": ef,     # Latin font (for English technical content & patient data)
    }


# ── Section builders ──────────────────────────────────────────────────────── #

def _hr(thin: bool = False):
    return HRFlowable(width="100%",
                      thickness=0.5 if thin else 1.5,
                      color=colors.HexColor("#0d2b52"))


def _section_header(text: str, s) -> list:
    return [Spacer(1, 3 * mm), Paragraph(text, s["section"]), _hr(thin=True), Spacer(1, 2 * mm)]


def _info_table(rows: list, s, font: str) -> Table:
    en_font = s.get("en_font", "NotoSans")
    data = []
    for row in rows:
        cells = []
        for label, value in zip(row[::2], row[1::2]):
            # Label: translated → native font; Value: patient data → Latin font for safe rendering
            cells.append(Paragraph(f"<b>{label}</b>", ParagraphStyle("lbl", fontName=font,    fontSize=9)))
            cells.append(Paragraph(str(value),         ParagraphStyle("val", fontName=en_font, fontSize=9)))
        while len(cells) < 4:
            cells.extend([Paragraph("", s["body"]), Paragraph("", s["body"])])
        data.append(cells)

    t = Table(data, colWidths=[38 * mm, 57 * mm, 38 * mm, 57 * mm])
    t.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [colors.HexColor("#eaf2fb"), colors.HexColor("#f4f8fc")]),
        ("GRID",    (0, 0), (-1, -1), 0.4, colors.HexColor("#b8d4ea")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",  (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _metric_table(rows: list, s, font: str) -> Table:
    # Metric names and values are always English technical content → use Latin font
    ef = s.get("en_font", "NotoSans")
    data = [[
        Paragraph(f"<b>{metric}</b>", ParagraphStyle("mk", fontName=ef, fontSize=9)),
        Paragraph(str(value),         ParagraphStyle("mv", fontName=ef, fontSize=9)),
    ] for metric, value in rows]
    t = Table(data, colWidths=[80 * mm, 100 * mm])
    t.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [colors.HexColor("#f0f0f0"), colors.HexColor("#fafafa")]),
        ("GRID",    (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("VALIGN",  (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


# ── Single-language story builder ─────────────────────────────────────────── #

def _build_story(phase, patient, doctor_name, results, original_image,
                 lang, report_id, report_date, s):
    """Build all PDF story elements for one language."""
    font = s["font"]
    story = []

    # ── Cover header ────────────────────────────────────────────────────────── #
    phase_label = _t(lang, f"report_phase{phase}")
    story.append(Paragraph(_t(lang, "report_title"), s["title"]))
    story.append(Paragraph(phase_label, s["subtitle"]))
    story.append(_hr())
    story.append(Spacer(1, 4 * mm))

    # ── Patient / Doctor info ──────────────────────────────────────────────── #
    story.append(_info_table([
        [_t(lang, "patient_name"), patient.get("name", "—"),
         _t(lang, "doctor"),       f"Dr. {doctor_name}"],
        [_t(lang, "patient_age"),  str(patient.get("age", "—")),
         _t(lang, "date"),         report_date],
        [_t(lang, "patient_gender"), patient.get("gender", "—"),
         _t(lang, "eye_examined"),   patient.get("eye", "—")],
        [_t(lang, "report_id"),    report_id,
         "Institution",            "IIT Bhubaneswar — Eye AI Lab"],
    ], s, font))
    story.append(Spacer(1, 4 * mm))
    story.append(_hr(thin=True))

    # ── Input fundus image ─────────────────────────────────────────────────── #
    story += _section_header("Fundus Image", s)
    story.append(_np_to_rl(original_image, max_w_mm=130, max_h_mm=140))
    story.append(Paragraph("Original fundus photograph", s["img_label"]))

    # ── SECTION 1 — DR GRADING ─────────────────────────────────────────────── #
    story += _section_header(_t(lang, "dr_grading"), s)
    drg      = results.get("drg", {})
    dr_grade = drg.get("grade", "Unknown")
    dr_level = drg.get("level", 0)

    severity_colors = ["#1a7a1a", "#d4a800", "#d47200", "#cc2200", "#8b0000"]
    grade_color = severity_colors[min(dr_level, 4)]

    ef = s["en_font"]
    story.append(Paragraph(
        f"<b>Classification:</b> "
        f'<font color="{grade_color}">{dr_grade}</font>  '
        f"(Level {dr_level}/4)",
        ParagraphStyle("drg", fontName=ef, fontSize=10, spaceAfter=3 * mm)
    ))
    dr_desc = DR_DESCRIPTIONS.get(dr_grade, "")
    if dr_desc:
        story.append(Paragraph(f"<b>Clinical Description:</b> {dr_desc}", s["body"]))

    story.append(_metric_table([
        ("DR Severity Level",  f"{dr_level} / 4 — {dr_grade}"),
        ("Grading Scale",      "ETDRS / International Clinical DR Disease Severity Scale"),
        ("Model",              "EfficientNet-B6 (fine-tuned, 5-class)"),
        ("Recommended Action", FOLLOW_UP.get(dr_level, "Consult ophthalmologist.")),
    ], s, font))

    # ── SECTION 2 — OD/OC SEGMENTATION ────────────────────────────────────── #
    story += _section_header(_t(lang, "odoc"), s)
    odoc        = results.get("odoc", {})
    overlay_img = odoc.get("overlay")
    raw_odoc    = odoc.get("raw")
    meas        = odoc.get("measurements", {})

    story.append(Paragraph(
        _t(lang, "odoc_color_key") + "  |  " + _t(lang, "quadrants"),
        s["small_gray"]
    ))
    story.append(Spacer(1, 2 * mm))

    if overlay_img is not None and raw_odoc is not None:
        story.append(_side_by_side(
            raw_odoc,    "Segmentation (Green=Disc, Blue=Cup)",
            overlay_img, "Overlay with quadrant lines (S/I/N/T)",
            s, max_half_mm=87,
        ))
    elif overlay_img is not None:
        story.append(_np_to_rl(overlay_img, max_w_mm=130, max_h_mm=150))
        story.append(Paragraph("Transparent overlay (S=Superior, I=Inferior, N=Nasal, T=Temporal)", s["img_label"]))

    if meas:
        vcdr = meas.get("vcdr", 0)
        hcdr = meas.get("hcdr", 0)
        isnt = meas.get("isnt_normal", True)

        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph("<b>Morphometric Measurements</b>", s["bold"]))
        story.append(_metric_table([
            ("Vertical CDR (vCDR)",       f"{vcdr:.3f}  —  {_cdr_interpretation(vcdr)}"),
            ("Horizontal CDR (hCDR)",     f"{hcdr:.3f}"),
            ("Area-based CDR",            f"{meas.get('area_cdr', 0):.3f}"),
            ("Optic Disc — Vertical Ø",   f"{meas.get('disc_vert_diam_px', '—')} px"),
            ("Optic Disc — Horizontal Ø", f"{meas.get('disc_horiz_diam_px', '—')} px"),
            ("Optic Cup — Vertical Ø",    f"{meas.get('cup_vert_diam_px', '—')} px"),
            ("Optic Cup — Horizontal Ø",  f"{meas.get('cup_horiz_diam_px', '—')} px"),
            ("Disc Area (px²)",           f"{meas.get('disc_area_px', '—')}"),
            ("Cup Area (px²)",            f"{meas.get('cup_area_px', '—')}"),
        ], s, font))

        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph("<b>Neuroretinal Rim Analysis — ISNT Rule</b>", s["bold"]))
        story.append(Paragraph(
            "Normal ISNT rule: Inferior rim ≥ Superior ≥ Nasal ≥ Temporal",
            s["small_gray"]
        ))
        story.append(_metric_table([
            ("Superior Rim",  f"{meas.get('rim_superior_pct', 0):.1f}% of total rim area"),
            ("Inferior Rim",  f"{meas.get('rim_inferior_pct', 0):.1f}% of total rim area"),
            ("Nasal Rim",     f"{meas.get('rim_nasal_pct', 0):.1f}% of total rim area"),
            ("Temporal Rim",  f"{meas.get('rim_temporal_pct', 0):.1f}% of total rim area"),
            ("ISNT Rule",     _t(lang, "isnt_normal" if isnt else "isnt_abnormal")),
        ], s, font))

    # ── SECTION 3 — LESION DETECTION ──────────────────────────────────────── #
    story += _section_header(_t(lang, "lesion"), s)
    lesion       = results.get("lesion", {})
    lesion_img   = lesion.get("image")
    lesion_blend = lesion.get("blend")

    story.append(Paragraph(_t(lang, "lesion_color_key"), s["small_gray"]))
    story.append(Spacer(1, 2 * mm))

    if lesion_img is not None and lesion_blend is not None:
        story.append(_side_by_side(
            lesion_img,   "Lesion segmentation (colour-coded by type)",
            lesion_blend, "Overlay on original fundus",
            s, max_half_mm=87,
        ))
    elif lesion_img is not None:
        story.append(_np_to_rl(lesion_img, max_w_mm=130, max_h_mm=150))
        story.append(Paragraph("Multi-lesion segmentation map", s["img_label"]))

    areas = lesion.get("areas", {})
    if areas:
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph("<b>Detected Lesion Areas (pixels²)</b>", s["bold"]))
        story.append(_metric_table([
            ("Hard Exudates (HE)",  f"{areas.get('hard_exudates', 0)} px²  — Lipid deposits from leaking vessels"),
            ("Haemorrhages (HEM)",  f"{areas.get('hemorrhages', 0)} px²  — Intraretinal bleeding"),
            ("Microaneurysms (MA)", f"{areas.get('microaneurysms', 0)} px²  — Small capillary bulges"),
            ("Soft Exudates (SE)",  f"{areas.get('soft_exudates', 0)} px²  — Cotton-wool spots"),
            ("Model",               "UNet + FIAM attention (IDRiD dataset)"),
        ], s, font))

    # ── SECTION 4 — GLAUCOMA GRADING ──────────────────────────────────────── #
    story += _section_header(_t(lang, "glaucoma"), s)
    glauc         = results.get("glaucoma", {})
    glaucoma_grade = glauc.get("grade", "Unknown")

    story.append(Paragraph(
        f"<b>Glaucoma Severity Grade:</b>  {glaucoma_grade}",
        ParagraphStyle("gl", fontName=ef, fontSize=10, spaceAfter=3 * mm)
    ))
    gl_desc = GLAUCOMA_DESCRIPTIONS.get(glaucoma_grade, "")
    if gl_desc:
        story.append(Paragraph(f"<b>Clinical Interpretation:</b> {gl_desc}", s["body"]))

    vcdr_val = meas.get("vcdr", 0) if meas else 0
    story.append(_metric_table([
        ("Model",               "SE-ResNet101 (3-class glaucoma grading)"),
        ("Classification",      glaucoma_grade),
        ("Corroborating CDR",   f"vCDR = {vcdr_val:.3f}  (suspicion threshold: ≥ 0.65)"),
        ("ISNT Rule Violation", "Yes — suggestive of glaucomatous damage"
                                 if not meas.get("isnt_normal", True) else "No"),
    ], s, font))

    # ── SECTION 5 — MICROANEURYSM (Phase 2 only) ──────────────────────────── #
    if phase == 2:
        ma       = results.get("ma", {})
        ma_img   = ma.get("image")
        ma_count = ma.get("count", 0)
        ma_orig  = ma.get("original_overlay")

        story += _section_header(_t(lang, "ma"), s)
        story.append(Paragraph(
            f"<b>{_t(lang, 'ma_count')}:</b>  {ma_count} cluster(s)",
            ParagraphStyle("ma", fontName=ef, fontSize=10, spaceAfter=3 * mm)
        ))

        if ma_img is not None and ma_orig is not None:
            story.append(_side_by_side(
                ma_orig, "Probability map (green overlay)",
                ma_img,  f"Detected MA clusters ({ma_count} shown as red circles)",
                s, max_half_mm=87,
            ))
        elif ma_img is not None:
            story.append(_np_to_rl(ma_img, max_w_mm=130, max_h_mm=150))
            story.append(Paragraph(f"Microaneurysm clusters ({ma_count} detected)", s["img_label"]))

        ma_risk = ("Low" if ma_count == 0 else
                   "Moderate" if ma_count <= 5 else
                   "High" if ma_count <= 15 else "Very High")
        story.append(Spacer(1, 2 * mm))
        story.append(_metric_table([
            ("MA Cluster Count",      str(ma_count)),
            ("Estimated Risk Level",  ma_risk),
            ("Clinical Significance", "MAs are the earliest ophthalmoscopic sign of DR. Count ≥ 5 warrants referral."),
            ("Detection Method",      "UNet + SimAM attention (patch-based, CLAHE green-channel)"),
        ], s, font))

    # ── SECTION 6 — CLINICAL SUMMARY ──────────────────────────────────────── #
    story.append(PageBreak())
    story += _section_header(_t(lang, "clinical_summary"), s)

    cdr_risk     = "Low" if vcdr_val < 0.5 else ("Moderate" if vcdr_val <= 0.7 else "High")
    ma_count_val = results.get("ma", {}).get("count", 0) if phase == 2 else "N/A (Phase 1)"

    summary_rows = [
        ("DR Grading",       f"{dr_grade}  (Level {dr_level}/4)"),
        ("Glaucoma Grading", glaucoma_grade),
        ("Vertical CDR",     f"{vcdr_val:.3f}  — Risk: {cdr_risk}"),
        ("ISNT Rule",        "Normal" if meas.get("isnt_normal", True) else "VIOLATED — suspicious"),
        ("Lesion Load",      f"HE: {areas.get('hard_exudates',0)} | HEM: {areas.get('hemorrhages',0)} | SE: {areas.get('soft_exudates',0)} px²"),
    ]
    if phase == 2:
        summary_rows.append(("Microaneurysm Count", f"{ma_count_val} cluster(s)"))
    story.append(_metric_table(summary_rows, s, font))

    story += _section_header(_t(lang, "recommendations"), s)
    rec_text = FOLLOW_UP.get(dr_level, "Consult ophthalmologist.")
    story.append(Paragraph(f"• DR follow-up: {rec_text}", s["body"]))
    if not meas.get("isnt_normal", True) or vcdr_val > 0.65:
        story.append(Paragraph(
            "• Optic disc: Formal glaucoma evaluation (IOP, visual fields, OCT-RNFL) recommended.",
            s["warn"]
        ))
    if areas.get("hemorrhages", 0) > 500 or areas.get("soft_exudates", 0) > 200:
        story.append(Paragraph(
            "• Lesion burden elevated — urgent ophthalmology review advised.",
            s["warn"]
        ))
    if phase == 2 and isinstance(ma_count_val, int) and ma_count_val >= 5:
        story.append(Paragraph(
            f"• {ma_count_val} MA clusters detected — exceeds referral threshold of 5. "
            f"Ophthalmology referral within 3 months.",
            s["warn"]
        ))

    # ── Footer ─────────────────────────────────────────────────────────────── #
    story.append(Spacer(1, 6 * mm))
    story.append(_hr())
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        f"Report ID: {report_id}  |  {report_date}",
        ParagraphStyle("foot_id", fontName=ef, fontSize=8,
                       textColor=colors.HexColor("#888"), alignment=TA_CENTER, spaceAfter=1 * mm)
    ))
    story.append(Paragraph(
        _t(lang, "generated_by"),
        ParagraphStyle("foot_by", fontName=font, fontSize=8,
                       textColor=colors.HexColor("#888"), alignment=TA_CENTER, spaceAfter=1 * mm)
    ))
    story.append(Paragraph(
        _t(lang, "disclaimer"),
        ParagraphStyle("disclaimer", fontName=font, fontSize=7,
                       textColor=colors.HexColor("#999"), alignment=TA_CENTER)
    ))

    return story


# ── Public API ────────────────────────────────────────────────────────────── #

def generate_report(
    phase: int,
    patient: dict,
    doctor_name: str,
    results: dict,
    original_image: np.ndarray,
    lang: str = "en",
    report_id: str | None = None,
) -> bytes:
    """
    Build and return PDF bytes.

    Structure:
      - Full report in English (always)
      - PageBreak + divider banner
      - Full report in selected language (only when lang != "en")
    """
    _register_fonts()
    if report_id is None:
        report_id = str(uuid.uuid4())[:8].upper()

    report_date = datetime.now().strftime("%d %B %Y, %I:%M %p")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
    )

    # English section (always first)
    en_s  = _make_styles("en")
    story = _build_story(phase, patient, doctor_name, results, original_image,
                         "en", report_id, report_date, en_s)

    # Native language section (appended after a clear page break + divider)
    if lang != "en":
        lang_s   = _make_styles(lang)
        lang_font = lang_s["font"]
        story.append(PageBreak())
        story.append(Spacer(1, 6 * mm))
        story.append(Paragraph(
            "─── Translation / अनुवाद / అనువాదం / மொழிபெயர்ப்பு ───",
            ParagraphStyle("divider",
                fontName="NotoSans", fontSize=11, leading=16,
                textColor=colors.HexColor("#0d2b52"),
                alignment=TA_CENTER, spaceBefore=2 * mm, spaceAfter=6 * mm)
        ))
        story += _build_story(phase, patient, doctor_name, results, original_image,
                              lang, report_id, report_date, lang_s)

    doc.build(story)
    buf.seek(0)
    return buf.read()
