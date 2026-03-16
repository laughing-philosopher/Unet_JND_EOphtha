"""
report.py — PDF report generation for Aakhi.
Generates a clean clinical report with patient info,
doctor info, input image and all model output images.
"""

import io
import os
from datetime import datetime

import numpy as np
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _np_to_rl_image(arr: np.ndarray, max_width_mm: float = 150) -> RLImage:
    """Convert a numpy image array to a ReportLab Image flowable."""
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    pil = PILImage.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)

    # Scale to fit max width while preserving aspect ratio
    max_w = max_width_mm * mm
    orig_w, orig_h = pil.size
    scale = max_w / orig_w
    return RLImage(buf, width=max_w, height=orig_h * scale)


def _pil_to_rl_image(pil_img: PILImage.Image, max_width_mm: float = 150) -> RLImage:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    max_w = max_width_mm * mm
    orig_w, orig_h = pil_img.size
    scale = max_w / orig_w
    return RLImage(buf, width=max_w, height=orig_h * scale)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def generate_report(
    patient_name: str,
    patient_age: int,
    patient_gender: str,
    doctor_name: str,
    model_title: str,
    input_image,           # PIL Image or np.ndarray
    output_images: list,   # list of (label: str, image: np.ndarray or PIL)
) -> bytes:
    """
    Build and return a PDF report as bytes.

    Parameters
    ----------
    patient_name    : str
    patient_age     : int
    patient_gender  : str
    doctor_name     : str
    model_title     : str  — e.g. "Microaneurysm Detector (MA)"
    input_image     : PIL Image or np.ndarray (RGB)
    output_images   : list of (label, image) tuples

    Returns
    -------
    bytes — the raw PDF content, ready for st.download_button
    """

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    W, H = A4
    content_width = W - 40 * mm

    # ---- Styles ----
    base = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=base["Title"],
        fontSize=20,
        textColor=colors.HexColor("#1a3a5c"),
        spaceAfter=2 * mm,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=base["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#555555"),
        alignment=TA_CENTER,
        spaceAfter=4 * mm,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=base["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#1a3a5c"),
        spaceBefore=5 * mm,
        spaceAfter=2 * mm,
        borderPad=2,
    )
    label_style = ParagraphStyle(
        "Label",
        parent=base["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#777777"),
        alignment=TA_CENTER,
        spaceAfter=1 * mm,
    )
    normal = base["Normal"]

    story = []

    # ---- Header bar ----
    story.append(Paragraph("Aakhi", title_style))
    story.append(Paragraph("Retinal Image Analysis Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a3a5c")))
    story.append(Spacer(1, 4 * mm))

    # ---- Info table (patient + doctor + date) ----
    report_date = datetime.now().strftime("%d %B %Y, %I:%M %p")

    info_data = [
        [
            Paragraph("<b>Patient Name</b>", normal), Paragraph(patient_name, normal),
            Paragraph("<b>Doctor</b>", normal),       Paragraph(f"Dr. {doctor_name}", normal),
        ],
        [
            Paragraph("<b>Age</b>", normal),          Paragraph(str(patient_age), normal),
            Paragraph("<b>Date</b>", normal),          Paragraph(report_date, normal),
        ],
        [
            Paragraph("<b>Gender</b>", normal),       Paragraph(patient_gender, normal),
            Paragraph("<b>Model Used</b>", normal),   Paragraph(model_title, normal),
        ],
    ]

    info_table = Table(info_data, colWidths=[35 * mm, 60 * mm, 35 * mm, 60 * mm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f8fc")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#eaf2fb"), colors.HexColor("#f4f8fc")]),
        ("GRID",      (0, 0), (-1, -1), 0.5, colors.HexColor("#c5d8ec")),
        ("PADDING",   (0, 0), (-1, -1), 5),
        ("VALIGN",    (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 5 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#c5d8ec")))

    # ---- Input image ----
    story.append(Paragraph("Input Fundus Image", section_style))
    if isinstance(input_image, np.ndarray):
        rl_input = _np_to_rl_image(input_image, max_width_mm=130)
    else:
        rl_input = _pil_to_rl_image(input_image, max_width_mm=130)

    story.append(KeepTogether([
        rl_input,
        Paragraph("Original uploaded fundus image", label_style),
    ]))

    # ---- Model outputs ----
    if output_images:
        story.append(Spacer(1, 4 * mm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#c5d8ec")))
        story.append(Paragraph("Model Output", section_style))

        for label, img in output_images:
            if img is None:
                continue
            if isinstance(img, np.ndarray):
                rl_img = _np_to_rl_image(img, max_width_mm=130)
            else:
                rl_img = _pil_to_rl_image(img, max_width_mm=130)

            story.append(KeepTogether([
                rl_img,
                Paragraph(label, label_style),
                Spacer(1, 3 * mm),
            ]))

    # ---- Footer ----
    story.append(Spacer(1, 6 * mm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a3a5c")))
    story.append(Paragraph(
        "This report is generated by Aakhi — Retinal Image Analysis. "
        "For clinical use only. Please consult a qualified ophthalmologist.",
        ParagraphStyle("Footer", parent=base["Normal"], fontSize=8,
                       textColor=colors.gray, alignment=TA_CENTER, spaceBefore=2 * mm),
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()