"""
report.py — PDF report generation for Aakhi.
Generates a clean clinical report with separate numbering for Figures and Tables.
"""

import io
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

def _np_to_rl_image(arr: np.ndarray, max_width_mm: float = 140) -> RLImage:
    """Convert a numpy image array to a ReportLab Image flowable."""
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    pil = PILImage.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)

    max_w = max_width_mm * mm
    orig_w, orig_h = pil.size
    scale = max_w / orig_w
    return RLImage(buf, width=max_w, height=orig_h * scale)


def _pil_to_rl_image(pil_img: PILImage.Image, max_width_mm: float = 140) -> RLImage:
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
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    base = getSampleStyleSheet()

    # ---- Custom Styles ----
    title_style = ParagraphStyle(
        "ReportTitle", parent=base["Title"], fontSize=22, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1c4b82"), alignment=TA_CENTER
    )
    section_style = ParagraphStyle(
        "Section", parent=base["Heading2"], fontSize=13, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1c4b82"), spaceBefore=8 * mm, textTransform="uppercase"
    )
    caption_style = ParagraphStyle(
        "Caption", parent=base["Normal"], fontSize=10, fontName="Helvetica-Oblique",
        textColor=colors.HexColor("#333333"), alignment=TA_CENTER, spaceBefore=2 * mm
    )
    # Special style for the DR Grading Result block
    dr_result_style = ParagraphStyle(
        "DRResult", parent=base["Normal"], fontSize=10, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1c4b82"), alignment=TA_CENTER
    )

    story = []
    fig_counter = 1  
    table_counter = 1 

    # ---- Header & Patient Info ----
    import os
    _BASE_DIR = os.environ.get('AAKHI_BASE_PATH') or os.path.dirname(os.path.abspath(__file__))
    logo_iitbbs_path = os.path.join(_BASE_DIR, "iitbbs logo.png")
    logo_aakhi_path = os.path.join(_BASE_DIR, "aakhi_logo.png")
    
    row = []
    if os.path.exists(logo_iitbbs_path):
        row.append(RLImage(logo_iitbbs_path, width=25*mm, height=25*mm, kind='proportional'))
    else:
        row.append("")
        
    title_p = Paragraph("AAKHI", title_style)
    subtitle_p = Paragraph("Comprehensive Retinal Image Analysis Report", ParagraphStyle("sub", parent=base["Italic"], alignment=TA_CENTER))
    row.append([title_p, subtitle_p])
    
    if os.path.exists(logo_aakhi_path):
        row.append(RLImage(logo_aakhi_path, width=25*mm, height=25*mm, kind='proportional'))
    else:
        row.append("")
        
    header_table = Table([row], colWidths=[35*mm, 100*mm, 35*mm])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (0,0), 'LEFT'),
        ('ALIGN', (1,0), (1,0), 'CENTER'),
        ('ALIGN', (2,0), (2,0), 'RIGHT'),
    ]))
    
    story.append(header_table)
    story.append(Spacer(1, 2*mm))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1c4b82")))
    
    report_date = datetime.now().strftime("%d %B %Y, %I:%M %p")
    info_data = [
        ["Patient Name:", patient_name, "Referring Doctor:", f"Dr. {doctor_name}"],
        ["Age / Gender:", f"{patient_age} / {patient_gender}", "Date:", report_date]
    ]
    t_info = Table(info_data, colWidths=[30*mm, 55*mm, 35*mm, 50*mm])
    t_info.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f0f4f8")),
        ('BACKGROUND', (2,0), (2,-1), colors.HexColor("#f0f4f8")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
    ]))
    story.append(Spacer(1, 5*mm))
    story.append(t_info)

    # ---- Primary Imaging ----
    story.append(Paragraph("1. Primary Imaging", section_style))
    img_flow = _np_to_rl_image(input_image) if isinstance(input_image, np.ndarray) else _pil_to_rl_image(input_image)
    story.append(KeepTogether([
        img_flow,
        Paragraph(f"<b>Figure {fig_counter}:</b> Original fundus image.", caption_style)
    ]))
    fig_counter += 1

    # ---- AI Findings ----
    if output_images:
        story.append(Paragraph("2. AI-Assisted Clinical Findings", section_style))
        
        for label, img in output_images:
            # Handle DR Grading as a text block result 
            if "DR Grading Result" in label:
                story.append(Spacer(1, 5*mm))
                res_box = Table([[Paragraph(label, dr_result_style)]], colWidths=[140*mm])
                res_box.setStyle(TableStyle([
                    ('BOX', (0,0), (-1,-1), 2, colors.HexColor("#1c4b82")),
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#eef4ff")),
                    ('PADDING', (0,0), (-1,-1), 15),
                ]))
                story.append(KeepTogether([res_box, Spacer(1, 5*mm)]))
                continue

            if img is None: continue

            # Determine if current item is a Table or Figure 
            is_table = any(kw in label for kw in ["Statistics", "Table", "Summary", "Legend"])
            
            rl_img = _np_to_rl_image(img) if isinstance(img, np.ndarray) else _pil_to_rl_image(img)
            
            if is_table:
                caption = f"<b>Table {table_counter}:</b> {label}"
                table_counter += 1
            else:
                caption = f"<b>Figure {fig_counter}:</b> {label}"
                fig_counter += 1

            story.append(KeepTogether([
                rl_img,
                Paragraph(caption, caption_style),
                Spacer(1, 6*mm)
            ]))

    # ---- Signature & Footer ----
    story.append(Spacer(1, 10*mm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
    story.append(Paragraph("<b>Physician Signature:</b> ___________________________", base["Normal"]))
    
    doc.build(story)
    buf.seek(0)
    return buf.read()