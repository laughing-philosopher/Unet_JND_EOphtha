"""overlay_odoc.py
=================
Creates a transparent ODOC segmentation overlay with 4 anatomical quadrants
(Superior / Inferior / Nasal / Temporal) and computes CDR measurements.

Input  : original fundus image (RGB uint8) + ODOC2 model output
         (RGB uint8 where Green=Disc rim, Blue=Cup)
Output : annotated overlay image + measurements dict
"""

import cv2
import numpy as np


def create_odoc_overlay(
    original_rgb: np.ndarray,
    odoc_output: np.ndarray,
    alpha: float = 0.38,
) -> tuple:
    """
    Returns (overlay_img, measurements_dict).

    overlay_img  : (H, W, 3) uint8 — original with semi-transparent disc/cup
                   colouring, quadrant cross-hairs, and boundary circles.
    measurements : dict with CDR values, pixel areas, quadrant rim percentages.
                   Empty dict if no disc was segmented.
    """
    h, w = original_rgb.shape[:2]

    # ── Extract masks ──────────────────────────────────────────────────────── #
    # ODOC2 colours: Green [0,255,0] = disc rim area, Blue [0,0,255] = cup
    disc_rim_mask = (odoc_output[:, :, 1] > 128) & (odoc_output[:, :, 2] < 128)
    cup_mask      = (odoc_output[:, :, 2] > 128) & (odoc_output[:, :, 1] < 128)
    full_disc     = disc_rim_mask | cup_mask   # complete disc region

    disc_area = int(np.sum(full_disc))
    cup_area  = int(np.sum(cup_mask))

    # ── Build semi-transparent overlay ────────────────────────────────────── #
    overlay = original_rgb.astype(np.float32).copy()
    if disc_area > 0:
        overlay[disc_rim_mask] = (
            overlay[disc_rim_mask] * (1 - alpha)
            + np.array([0, 210, 0], dtype=np.float32) * alpha
        )
        overlay[cup_mask] = (
            overlay[cup_mask] * (1 - alpha)
            + np.array([60, 60, 255], dtype=np.float32) * alpha
        )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    if disc_area == 0:
        return overlay, {}

    # ── Contours ──────────────────────────────────────────────────────────── #
    disc_bin = full_disc.astype(np.uint8) * 255
    cup_bin  = cup_mask.astype(np.uint8)  * 255

    disc_cnts, _ = cv2.findContours(disc_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cup_cnts,  _ = cv2.findContours(cup_bin,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not disc_cnts:
        return overlay, {}

    (cx, cy), disc_r = cv2.minEnclosingCircle(max(disc_cnts, key=cv2.contourArea))
    cx, cy, disc_r = int(cx), int(cy), int(disc_r)

    cup_r = 0
    if cup_cnts:
        _, _cr = cv2.minEnclosingCircle(max(cup_cnts, key=cv2.contourArea))
        cup_r = int(_cr)

    # ── CDR calculations ──────────────────────────────────────────────────── #
    disc_rows = np.where(np.any(full_disc, axis=1))[0]
    cup_rows  = np.where(np.any(cup_mask,  axis=1))[0]
    disc_cols = np.where(np.any(full_disc, axis=0))[0]
    cup_cols  = np.where(np.any(cup_mask,  axis=0))[0]

    disc_vert  = int(disc_rows[-1] - disc_rows[0]) if len(disc_rows) > 1 else 1
    cup_vert   = int(cup_rows[-1]  - cup_rows[0])  if len(cup_rows)  > 1 else 0
    disc_horiz = int(disc_cols[-1] - disc_cols[0]) if len(disc_cols) > 1 else 1
    cup_horiz  = int(cup_cols[-1]  - cup_cols[0])  if len(cup_cols)  > 1 else 0

    vcdr     = round(cup_vert  / disc_vert,  3) if disc_vert  > 0 else 0.0
    hcdr     = round(cup_horiz / disc_horiz, 3) if disc_horiz > 0 else 0.0
    area_cdr = round(float(np.sqrt(cup_area / disc_area)), 3) if disc_area > 0 else 0.0

    # ── Quadrant rim analysis (ISNT rule) ─────────────────────────────────── #
    total_rim = max(disc_area - cup_area, 1)

    def _rim_pct(r0, r1, c0, c1):
        d = int(np.sum(full_disc[r0:r1, c0:c1]))
        c = int(np.sum(cup_mask[r0:r1,  c0:c1]))
        return round((d - c) / total_rim * 100, 1)

    rim_sup  = _rim_pct(0,  cy, 0,  w)
    rim_inf  = _rim_pct(cy, h,  0,  w)
    rim_nas  = _rim_pct(0,  h,  0,  cx)   # Left side = Nasal for right eye (approx.)
    rim_temp = _rim_pct(0,  h,  cx, w)

    # ISNT rule: normal if Inferior >= Superior >= Nasal >= Temporal
    isnt_normal = (rim_inf >= rim_sup) and (rim_sup >= rim_nas) and (rim_nas >= rim_temp)

    measurements = {
        "disc_center":       (cx, cy),
        "disc_radius_px":    disc_r,
        "cup_radius_px":     cup_r,
        "vcdr":              vcdr,
        "hcdr":              hcdr,
        "area_cdr":          area_cdr,
        "disc_area_px":      disc_area,
        "cup_area_px":       cup_area,
        "disc_vert_diam_px": disc_vert,
        "cup_vert_diam_px":  cup_vert,
        "disc_horiz_diam_px":disc_horiz,
        "cup_horiz_diam_px": cup_horiz,
        "rim_superior_pct":  rim_sup,
        "rim_inferior_pct":  rim_inf,
        "rim_nasal_pct":     rim_nas,
        "rim_temporal_pct":  rim_temp,
        "isnt_normal":       isnt_normal,
    }

    # ── Draw annotations on overlay ───────────────────────────────────────── #
    lw = max(1, disc_r // 30)

    # Disc boundary (green)
    cv2.circle(overlay, (cx, cy), disc_r, (0, 220, 0), lw)
    # Cup boundary (blue)
    if cup_r > 0:
        cv2.circle(overlay, (cx, cy), cup_r, (80, 80, 255), lw)

    # Quadrant cross-hairs (yellow)
    qc = (255, 220, 30)
    top    = max(0,     cy - disc_r)
    bottom = min(h - 1, cy + disc_r)
    left   = max(0,     cx - disc_r)
    right  = min(w - 1, cx + disc_r)
    cv2.line(overlay, (cx, top),  (cx, bottom), qc, lw)
    cv2.line(overlay, (left, cy), (right, cy),  qc, lw)

    # Quadrant labels
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fs     = max(0.35, disc_r / 80.0)
    ft     = max(1, lw)
    off    = max(12, disc_r // 3)
    labels = [
        ("S", (cx - ft * 5, cy - off)),
        ("I", (cx - ft * 4, cy + off + ft * 8)),
        ("N", (cx - off - ft * 10, cy + ft * 5)),
        ("T", (cx + off,            cy + ft * 5)),
    ]
    for txt, pos in labels:
        cv2.putText(overlay, txt, pos, font, fs, (255, 255, 0), ft, cv2.LINE_AA)

    return overlay, measurements
