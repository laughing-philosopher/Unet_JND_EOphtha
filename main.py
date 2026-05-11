"""main.py — Aakhi Flask application
=====================================
Unified retinal image analysis: DR Grading, OD/OC, Lesions, Glaucoma, MA
Two-phase report (Phase 1 fast, Phase 2 with MA).
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import threading
import time
import traceback
import uuid
from queue import Queue

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, redirect, render_template,
                   request, send_file, session, url_for)
from PIL import Image

from auth import verify_user, get_user, register_user

# ── Path helpers ─────────────────────────────────────────────────────────── #
# AAKHI_BASE_PATH is set by aakhi_hook.py when running as a PyInstaller bundle
# (points to sys._MEIPASS / _internal/). Falls back to the script's own directory.
BASE_DIR = (os.environ.get('AAKHI_BASE_PATH') or
            os.path.dirname(os.path.abspath(__file__)))


def _path(*parts):
    return os.path.join(BASE_DIR, *parts)


# ── Flask setup ───────────────────────────────────────────────────────────── #
app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))
app.secret_key = os.environ.get("AAKHI_SECRET", "aakhi-retinal-secret-2024")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload limit

# ── In-memory job store ───────────────────────────────────────────────────── #
#   jobs[job_id] = {
#     patient, status, results,
#     phase1_ready, phase2_ready,
#     ma_progress, ma_total,
#     error, sse_queue
#   }
jobs: dict[str, dict] = {}


# ── Lazy model loading (singletons) ──────────────────────────────────────── #
_dr_model        = None
_glaucoma_model  = None


def _get_dr_model():
    global _dr_model
    if _dr_model is None:
        from processing.processing_dr_grading import load_dr_model
        _dr_model = load_dr_model()
    return _dr_model


def _get_glaucoma_model():
    global _glaucoma_model
    if _glaucoma_model is None:
        from processing.processing_glaucoma_grading import load_glaucoma_model
        _glaucoma_model = load_glaucoma_model()
    return _glaucoma_model


# ── Image utilities ───────────────────────────────────────────────────────── #

def _np_to_b64(arr: np.ndarray) -> str:
    """Convert numpy RGB array to base64-encoded PNG string."""
    if arr is None:
        return ""
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _save_temp_image(img_array: np.ndarray) -> str:
    """Save numpy RGB array as temp PNG, return path."""
    pil = Image.fromarray(img_array.astype(np.uint8))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    pil.save(tmp.name)
    tmp.close()
    return tmp.name


def _classical_lesion_detection(img_rgb: np.ndarray) -> tuple:
    """
    Classical CV fallback for lesion detection when the FIAM model is unavailable.
    Returns (colour_coded_rgb, model_available=False).

    Detects:
      Red   → Hard Exudates  (bright yellowish-white regions, high L in LAB)
      Green → Haemorrhages   (dark reddish blobs)
      Blue  → (not detected classically)
      Yellow→ Soft Exudates  (grey/pale regions near vessels)
    """
    h, w = img_rgb.shape[:2]
    out  = np.zeros((h, w, 3), dtype=np.uint8)

    # Work in LAB colour space for better separation
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, A, B_ch = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # Green channel and CLAHE
    green_raw = img_rgb[:, :, 1].astype(np.float32)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_eq  = clahe.apply(img_rgb[:, :, 1])

    # ── Hard exudates: bright, high L, yellowish (B_ch > mean) ────────────── #
    l_thresh  = int(np.percentile(L, 85))
    he_mask   = (L > l_thresh) & (B_ch > 130) & (A < 140)
    # Remove very large regions (likely optic disc)
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    he_mask   = cv2.morphologyEx(he_mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    he_mask   = cv2.morphologyEx(he_mask, cv2.MORPH_DILATE, kernel, iterations=1)
    out[he_mask > 0] = [255, 0, 0]   # Red

    # ── Haemorrhages: dark red blobs (low L, higher A channel) ────────────── #
    l_dark  = int(np.percentile(L, 20))
    hem_mask= (L < l_dark) & (A > 128) & (img_rgb[:, :, 0] > img_rgb[:, :, 1])
    hem_mask= cv2.morphologyEx(hem_mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    out[hem_mask > 0] = [0, 255, 0]  # Green

    # ── Soft exudates: pale grey cotton-wool spots (moderate L, low saturation) #
    hsv      = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    S        = hsv[:, :, 1]
    se_mask  = (L > int(np.percentile(L, 70))) & (S < 60) & (he_mask == 0)
    se_mask  = cv2.morphologyEx(se_mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    out[se_mask > 0] = [255, 255, 0]  # Yellow

    return out, False


def _cdr_refined_glaucoma(raw_grade: str, measurements: dict) -> tuple:
    """
    Refine glaucoma grade using CDR measurements and ISNT rule from ODOC.

    Clinical rules applied (evidence-based):
      vCDR >= 0.80              → at least Moderate Glaucoma
      vCDR >= 0.65              → at least Glaucoma Suspect
      ISNT rule violated        → at least Glaucoma Suspect
      vCDR >= 0.65 + ISNT bad   → upgrade one level beyond raw grade

    Returns (refined_grade: str, reason: str)
    """
    GRADES = ["No Glaucoma", "Glaucoma Suspect", "Moderate Glaucoma", "Advanced Glaucoma"]
    idx    = GRADES.index(raw_grade) if raw_grade in GRADES else 0

    vcdr       = measurements.get("vcdr", 0.0)
    isnt_ok    = measurements.get("isnt_normal", True)
    reason     = "image model prediction"

    # Apply CDR-based escalation
    if vcdr >= 0.80:
        new_idx = max(idx, 2)  # at least Moderate
        reason  = f"vCDR={vcdr:.3f} >= 0.80 (high glaucoma probability threshold)"
    elif vcdr >= 0.65 and not isnt_ok:
        new_idx = max(idx, max(1, idx + 1))  # upgrade one step
        reason  = f"vCDR={vcdr:.3f} >= 0.65 + ISNT rule violated"
    elif vcdr >= 0.65:
        new_idx = max(idx, 1)  # at least Suspect
        reason  = f"vCDR={vcdr:.3f} >= 0.65 (borderline CDR)"
    elif not isnt_ok:
        new_idx = max(idx, 1)  # at least Suspect
        reason  = "ISNT rule violated (neuroretinal rim asymmetry)"
    else:
        new_idx = idx

    new_idx = min(new_idx, len(GRADES) - 1)
    refined  = GRADES[new_idx]
    if refined != raw_grade:
        reason = f"Upgraded from '{raw_grade}': {reason}"

    return refined, reason


def _lesion_areas(lesion_rgb: np.ndarray) -> dict:
    """Count pixel areas for each lesion class from colour-coded output."""
    r, g, b = lesion_rgb[:, :, 0], lesion_rgb[:, :, 1], lesion_rgb[:, :, 2]
    return {
        "hard_exudates": int(np.sum((r > 128) & (g < 64) & (b < 64))),
        "hemorrhages":   int(np.sum((g > 128) & (r < 64) & (b < 64))),
        "microaneurysms":int(np.sum((b > 128) & (r < 64) & (g < 64))),
        "soft_exudates": int(np.sum((r > 128) & (g > 128) & (b < 64))),
    }


def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ── Analysis pipeline ─────────────────────────────────────────────────────── #

def _run_phase1(job_id: str, img_rgb: np.ndarray) -> None:
    """Run all fast models synchronously; update job state."""
    job = jobs[job_id]
    q: Queue = job["sse_queue"]

    def _update(msg: str):
        job["status_msg"] = msg
        q.put({"type": "status", "msg": msg})

    results = job.setdefault("results", {})

    # ── DR Grading ────────────────────────────────────────────────────────── #
    try:
        _update("Running DR Grading...")
        from processing.processing_dr_grading import predict_dr_severity
        tmp = _save_temp_image(img_rgb)
        grade = predict_dr_severity(tmp, _get_dr_model())
        os.unlink(tmp)
        dr_levels = {
            "No DR": 0, "Mild DR": 1, "Moderate DR": 2,
            "Severe DR": 3, "Proliferative DR": 4,
        }
        results["drg"] = {"grade": grade, "level": dr_levels.get(grade, 0)}
    except Exception as e:
        results["drg"] = {"grade": "Error", "level": 0, "error": str(e)}
        traceback.print_exc()

    q.put({"type": "progress", "step": "drg", "done": True})

    # ── OD/OC Segmentation ────────────────────────────────────────────────── #
    try:
        _update("Running OD/OC Segmentation...")
        from processing.processing_odoc import processing as odoc_proc

        # processing_odoc returns (outlined_img, features_dict, rf_label, rf_prob, seg_mask)
        outlined_img, odoc_features, rf_pred_label, rf_prob, full_pred = odoc_proc(img_rgb, threshold=0.5, batch_size=8)

        # Build colour-coded ODOC image for overlay_odoc.py
        #   Green [0,255,0] = disc rim area, Blue [0,0,255] = cup
        orig_h, orig_w = img_rgb.shape[:2]
        raw_odoc = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        raw_odoc[full_pred == 1] = [0, 255, 0]   # Green = disc rim
        raw_odoc[full_pred == 2] = [0, 0, 255]    # Blue  = cup

        from processing.overlay_odoc import create_odoc_overlay
        overlay, meas = create_odoc_overlay(img_rgb, raw_odoc, alpha=0.38)

        # Store RF glaucoma prediction for later use
        results["odoc_rf"] = {
            "features":  odoc_features,
            "rf_pred":   rf_pred_label,
            "rf_prob":   rf_prob,
        }
        results["odoc"] = {
            "overlay_b64":  _np_to_b64(overlay),
            "raw_b64":      _np_to_b64(raw_odoc),
            "measurements": meas,
            "overlay_arr":  overlay,
            "raw_arr":      raw_odoc,
        }
    except Exception as e:
        results["odoc"] = {"overlay_b64": "", "raw_b64": "", "measurements": {}, "error": str(e)}
        traceback.print_exc()

    q.put({"type": "progress", "step": "odoc", "done": True})

    # ── Multi-Lesion Detection ────────────────────────────────────────────── #
    try:
        _update("Running Lesion Detection...")
        lesion_model_path = _path("models", "Unet+FIAM_IDriD_70epochs_1.2_300.h5")
        if os.path.exists(lesion_model_path):
            from processing.processing_lesion import processing as lesion_proc
            lesion_out = np.array(lesion_proc(img_rgb, threshold=0.5, batch_size=1), dtype=np.uint8)
            model_available = True
        else:
            # Classical CV fallback: detect bright exudates and dark hemorrhages
            lesion_out, model_available = _classical_lesion_detection(img_rgb)

        blend = cv2.addWeighted(img_rgb, 0.55, lesion_out, 0.45, 0)
        areas = _lesion_areas(lesion_out)
        results["lesion"] = {
            "image_b64":       _np_to_b64(lesion_out),
            "blend_b64":       _np_to_b64(blend),
            "areas":           areas,
            "image_arr":       lesion_out,
            "blend_arr":       blend,
            "model_available": model_available,
        }
    except Exception as e:
        results["lesion"] = {"image_b64": "", "blend_b64": "", "areas": {},
                             "model_available": False, "error": str(e)}
        traceback.print_exc()

    q.put({"type": "progress", "step": "lesion", "done": True})

    # ── Glaucoma Grading (RF Model-based) ─────────────────────────────────── #
    try:
        _update("Running Glaucoma Grading...")
        
        # Use RF model output generated during OD/OC step
        rf_pred = results.get("odoc_rf", {}).get("rf_pred", "Unknown")
        rf_prob = results.get("odoc_rf", {}).get("rf_prob", 0.0)
        
        if rf_pred == "Glaucoma":
            if rf_prob >= 0.7:
                gl_grade = "Advanced Glaucoma"
            elif rf_prob >= 0.5:
                gl_grade = "Moderate Glaucoma"
            else:
                gl_grade = "Glaucoma Suspect"
            cdr_reason = f"Random Forest detected Glaucoma (Probability: {rf_prob:.1%})"
        elif rf_pred == "Normal":
            gl_grade = "No Glaucoma"
            cdr_reason = f"Random Forest predicts Normal (Probability: {(1-rf_prob):.1%})"
        else:
            gl_grade = "Error"
            cdr_reason = "RF Model failed or OD/OC extraction failed."

        raw_grade = f"RF: {rf_pred} (Prob: {rf_prob:.2f})"

        gl_levels = {
            "No Glaucoma": 0, "Glaucoma Suspect": 1,
            "Moderate Glaucoma": 2, "Advanced Glaucoma": 3,
        }
        results["glaucoma"] = {
            "grade":       gl_grade,
            "level":       gl_levels.get(gl_grade, 0),
            "raw_grade":   raw_grade,
            "cdr_reason":  cdr_reason,
        }
    except Exception as e:
        results["glaucoma"] = {"grade": "Error", "level": 0, "error": str(e)}
        traceback.print_exc()

    q.put({"type": "progress", "step": "glaucoma", "done": True})

    job["phase1_ready"] = True
    _update("Phase 1 complete")
    q.put({"type": "phase1_ready"})


def _build_circled_overlay(img_rgb: np.ndarray, binary_mask: np.ndarray):
    """Build a circled-cluster overlay from a binary mask. Returns (circled_img, count)."""
    bm = (binary_mask > 0).astype(np.uint8) * 255
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    dilated = cv2.dilate(bm, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circled = img_rgb.copy()
    for cnt in contours:
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cv2.circle(circled, (int(cx), int(cy)), max(int(radius), 10), (255, 255, 255), 2)
    return circled, len(contours)


def _build_removed_fp_overlay(img_rgb: np.ndarray, removed_locs: list):
    """Draw FP markers on image for removed false-positive locations."""
    out = img_rgb.copy()
    for (x, y) in removed_locs:
        cv2.circle(out, (x, y), 12, (255, 0, 0), 2)
        cv2.putText(out, "FP", (x + 10, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return out


def _run_ma(job_id: str, img_rgb: np.ndarray) -> None:
    """Run MA detection asynchronously; update progress via SSE queue."""
    job = jobs[job_id]
    q: Queue = job["sse_queue"]

    try:
        from processing.processing_ma import processing as ma_proc

        ma_model_path = _path("models", "UNet_JND_EOphtha.h5")
        if not os.path.exists(ma_model_path):
            raise FileNotFoundError(f"MA model not found: {ma_model_path}")

        total_batches = [0]

        def _progress(current: int, total: int):
            total_batches[0] = total
            job["ma_progress"] = current
            job["ma_total"]    = total
            pct = int(current / total * 100) if total > 0 else 0
            q.put({"type": "ma_progress", "current": current, "total": total, "pct": pct})

        q.put({"type": "ma_started"})
        result = ma_proc(img_rgb, threshold=0.1, batch_size=20, progress_callback=_progress)

        # result is now a dict with both baseline and refined masks
        prob_map      = result["prob_map"]
        baseline_mask = result["baseline_mask"]
        refined_mask  = result["refined_mask"]
        adv_count     = result["baseline_count"]
        basic_count   = result["refined_count"]
        removed_count = result["removed_count"]
        removed_locs  = result["removed_locations"]
        fp_pct        = (100.0 * removed_count / max(adv_count, 1))

        # Build circle overlays for both modes
        circled_adv,   n_adv   = _build_circled_overlay(img_rgb, baseline_mask)
        circled_basic, n_basic = _build_circled_overlay(img_rgb, refined_mask)
        removed_vis            = _build_removed_fp_overlay(img_rgb, removed_locs)

        # Build green probability overlay (shared)
        disp  = (prob_map * 255.0).clip(0, 255).astype(np.uint8)
        green = np.zeros_like(img_rgb)
        green[:, :, 1] = disp
        prob_overlay = cv2.addWeighted(img_rgb, 0.7, green, 0.3, 0)

        jobs[job_id]["results"]["ma"] = {
            # Backward-compatible keys
            "image_b64":      _np_to_b64(circled_adv),
            "overlay_b64":    _np_to_b64(prob_overlay),
            "count":          n_adv,
            "image_arr":      circled_adv,
            "overlay_arr":    prob_overlay,
            # New Advanced / Basic keys
            "adv_count":        n_adv,
            "basic_count":      n_basic,
            "removed_count":    removed_count,
            "fp_reduction_pct": fp_pct,
            "adv_image_arr":    circled_adv,
            "basic_image_arr":  circled_basic,
            "removed_image_arr":removed_vis,
            "adv_image_b64":    _np_to_b64(circled_adv),
            "basic_image_b64":  _np_to_b64(circled_basic),
            "removed_image_b64":_np_to_b64(removed_vis),
        }
        jobs[job_id]["phase2_ready"] = True
        q.put({"type": "phase2_ready", "ma_count": n_adv, "ma_basic_count": n_basic})

    except Exception as e:
        traceback.print_exc()
        jobs[job_id]["results"]["ma"] = {"error": str(e), "count": 0, "adv_count": 0, "basic_count": 0}
        jobs[job_id]["phase2_ready"] = True  # MA failed but don't block phase2
        q.put({"type": "phase2_ready", "ma_count": 0, "ma_basic_count": 0})


def _run_rfnld(job_id: str, img_rgb: np.ndarray) -> None:
    """Run RFNLD detection asynchronously after MA; uses ODOC coordinates."""
    job = jobs[job_id]
    q: Queue = job["sse_queue"]

    try:
        from processing.processing_rfnld import processing as rfnld_proc
        from processing.processing_rfnld import derive_coordinates_from_odoc

        rfnld_model_path = _path("models", "retinet_9010.h5")
        if not os.path.exists(rfnld_model_path):
            raise FileNotFoundError(f"RFNLD model not found: {rfnld_model_path}")

        # Derive optic disc coordinates from ODOC segmentation
        odoc_meas = job["results"].get("odoc", {}).get("measurements", {})
        coords, coords2 = derive_coordinates_from_odoc(odoc_meas)

        if coords is None or coords2 is None:
            raise ValueError(
                "Could not derive optic disc coordinates from ODOC results. "
                "ODOC segmentation may have failed to detect the disc."
            )

        q.put({"type": "rfnld_started"})
        job["status_msg"] = "Running RFNLD Detection..."
        q.put({"type": "status", "msg": "Running RFNLD Detection..."})

        result = rfnld_proc(img_rgb, coords, coords2)

        rfnld_img    = result["image"]
        defects_found = result["defects_found"]
        defect_count  = result["defect_count"]

        jobs[job_id]["results"]["rfnld"] = {
            "image_b64":     _np_to_b64(rfnld_img),
            "image_arr":     rfnld_img,
            "defects_found": defects_found,
            "defect_count":  defect_count,
        }
        jobs[job_id]["rfnld_ready"] = True
        q.put({"type": "rfnld_ready", "defect_count": defect_count, "defects_found": defects_found})

    except Exception as e:
        traceback.print_exc()
        jobs[job_id]["results"]["rfnld"] = {
            "error": str(e), "defects_found": False, "defect_count": 0,
        }
        jobs[job_id]["rfnld_ready"] = True  # mark ready even on error
        q.put({"type": "rfnld_error", "msg": str(e)})


# ── Auth helpers ──────────────────────────────────────────────────────────── #

def _logged_in() -> bool:
    return session.get("logged_in", False) or session.get("guest_mode", False)


def _current_user() -> dict:
    if session.get("guest_mode"):
        return {"username": "guest", "full_name": "Guest User", "role": "guest"}
    return {
        "username":  session.get("username", ""),
        "full_name": session.get("full_name", ""),
        "role":      session.get("role", ""),
    }


# ── Routes ────────────────────────────────────────────────────────────────── #

@app.route("/")
def index():
    if not _logged_in():
        return redirect(url_for("login"))
    return render_template("index.html", user=_current_user())


@app.route("/login", methods=["GET", "POST"])
def login():
    if _logged_in():
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form
        username = str(data.get("username", "")).strip()
        password = str(data.get("password", ""))

        if username == "guest":
            session["guest_mode"]  = True
            session["logged_in"]   = False
            if request.is_json:
                return jsonify({"ok": True})
            return redirect(url_for("index"))

        user = get_user(username)
        if user and verify_user(username, password):
            session["logged_in"]  = True
            session["guest_mode"] = False
            session["username"]   = username
            session["full_name"]  = user.get("full_name", username)
            session["role"]       = user.get("role", "doctor")
            if request.is_json:
                return jsonify({"ok": True})
            return redirect(url_for("index"))

        error = "Invalid credentials."
        if request.is_json:
            return jsonify({"ok": False, "error": error}), 401

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ── Analysis API ──────────────────────────────────────────────────────────── #

@app.post("/api/analyze")
def api_analyze():
    if not _logged_in():
        return jsonify({"error": "Not authenticated"}), 401

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    f = request.files["image"]
    patient = {
        "name":   request.form.get("patient_name", "Unknown"),
        "age":    request.form.get("patient_age", "—"),
        "gender": request.form.get("patient_gender", "—"),
        "eye":    request.form.get("patient_eye", "—"),
    }

    # Decode image
    try:
        file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return jsonify({"error": f"Could not decode image: {e}"}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "patient":       patient,
        "image_arr":     img_rgb,
        "image_b64":     _np_to_b64(img_rgb),
        "status":        "running",
        "status_msg":    "Starting...",
        "results":       {},
        "phase1_ready":  False,
        "phase2_ready":  False,
        "rfnld_ready":   False,
        "ma_progress":   0,
        "ma_total":      1,
        "sse_queue":     Queue(),
        "user":          _current_user(),
    }

    # Phase 1 runs in a thread, then MA, then RFNLD
    def _phase1_then_phase2():
        _run_phase1(job_id, img_rgb)
        _run_ma(job_id, img_rgb)
        _run_rfnld(job_id, img_rgb)
        jobs[job_id]["status"] = "complete"

    threading.Thread(target=_phase1_then_phase2, daemon=True).start()

    return jsonify({"job_id": job_id})


@app.route("/api/stream/<job_id>")
def api_stream(job_id: str):
    """SSE endpoint streaming progress events."""
    if job_id not in jobs:
        return Response("data: {\"type\": \"error\", \"msg\": \"Job not found\"}\n\n",
                        content_type="text/event-stream")

    def _generate():
        q: Queue = jobs[job_id]["sse_queue"]
        while True:
            try:
                event = q.get(timeout=30)
                yield _sse_event(event)
                if event.get("type") in ("rfnld_ready", "rfnld_error",
                                          "complete", "error"):
                    break
                if event.get("type") in ("phase1_ready", "phase2_ready",
                                          "ma_error"):
                    # Keep streaming for MA + RFNLD progress
                    pass
            except Exception:
                yield _sse_event({"type": "heartbeat"})
                if jobs[job_id].get("rfnld_ready") or jobs[job_id]["status"] == "complete":
                    break

    return Response(_generate(),
                    content_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/results/<job_id>")
def api_results(job_id: str):
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404
    job = jobs[job_id]
    res = job.get("results", {})

    payload = {
        "status":        job["status"],
        "phase1_ready":  job["phase1_ready"],
        "phase2_ready":  job["phase2_ready"],
        "status_msg":    job.get("status_msg", ""),
        "ma_progress":   job.get("ma_progress", 0),
        "ma_total":      job.get("ma_total", 1),
        "patient":       job["patient"],
        "original_b64":  job.get("image_b64", ""),
        "results": {
            "drg": {
                "grade": res.get("drg", {}).get("grade", "—"),
                "level": res.get("drg", {}).get("level", 0),
            },
            "odoc": {
                "overlay_b64":  res.get("odoc", {}).get("overlay_b64", ""),
                "raw_b64":      res.get("odoc", {}).get("raw_b64", ""),
                "measurements": res.get("odoc", {}).get("measurements", {}),
                "error":        res.get("odoc", {}).get("error", ""),
            },
            "lesion": {
                "image_b64":       res.get("lesion", {}).get("image_b64", ""),
                "blend_b64":       res.get("lesion", {}).get("blend_b64", ""),
                "areas":           res.get("lesion", {}).get("areas", {}),
                "model_available": res.get("lesion", {}).get("model_available", True),
                "error":           res.get("lesion", {}).get("error", ""),
            },
            "glaucoma": {
                "grade": res.get("glaucoma", {}).get("grade", "—"),
                "level": res.get("glaucoma", {}).get("level", 0),
            },
            "ma": {
                "status":       "complete" if job["phase2_ready"] else
                                ("running" if job["phase1_ready"] else "pending"),
                "count":            res.get("ma", {}).get("count", 0),
                "adv_count":        res.get("ma", {}).get("adv_count", 0),
                "basic_count":      res.get("ma", {}).get("basic_count", 0),
                "removed_count":    res.get("ma", {}).get("removed_count", 0),
                "fp_reduction_pct": res.get("ma", {}).get("fp_reduction_pct", 0.0),
                "image_b64":        res.get("ma", {}).get("image_b64", ""),
                "adv_image_b64":    res.get("ma", {}).get("adv_image_b64", ""),
                "basic_image_b64":  res.get("ma", {}).get("basic_image_b64", ""),
                "removed_image_b64":res.get("ma", {}).get("removed_image_b64", ""),
                "overlay_b64":      res.get("ma", {}).get("overlay_b64", ""),
                "error":            res.get("ma", {}).get("error", ""),
                "progress_pct":     int(job["ma_progress"] / max(job["ma_total"], 1) * 100),
            },
            "rfnld": {
                "status":        "complete" if job.get("rfnld_ready") else
                                 ("running" if job["phase2_ready"] else "pending"),
                "image_b64":     res.get("rfnld", {}).get("image_b64", ""),
                "defects_found": res.get("rfnld", {}).get("defects_found", False),
                "defect_count":  res.get("rfnld", {}).get("defect_count", 0),
                "error":         res.get("rfnld", {}).get("error", ""),
            },
        },
        "rfnld_ready":   job.get("rfnld_ready", False),
    }
    return jsonify(payload)


# ── Report API ────────────────────────────────────────────────────────────── #

@app.post("/api/report/<job_id>/<int:phase>")
def api_report(job_id: str, phase: int):
    if not _logged_in():
        return jsonify({"error": "Not authenticated"}), 401
    if job_id not in jobs:
        return jsonify({"error": "Job not found"}), 404

    job = jobs[job_id]
    if phase == 1 and not job["phase1_ready"]:
        return jsonify({"error": "Phase 1 not ready"}), 400
    if phase == 2 and not job["phase2_ready"]:
        return jsonify({"error": "Phase 2 not ready"}), 400

    lang       = request.json.get("lang", "en") if request.is_json else "en"
    report_id  = str(uuid.uuid4())[:8].upper()
    res        = job["results"]

    from report_v2 import generate_report

    pdf_bytes = generate_report(
        phase        = phase,
        patient      = job["patient"],
        doctor_name  = job["user"].get("full_name", "—"),
        results      = {
            "drg":      res.get("drg",      {}),
            "odoc":     {
                "overlay":      res.get("odoc", {}).get("overlay_arr"),
                "raw":          res.get("odoc", {}).get("raw_arr"),
                "measurements": res.get("odoc", {}).get("measurements", {}),
            },
            "lesion":   {
                "image":  res.get("lesion", {}).get("image_arr"),
                "blend":  res.get("lesion", {}).get("blend_arr"),
                "areas":  res.get("lesion", {}).get("areas", {}),
            },
            "glaucoma": res.get("glaucoma", {}),
            "ma":       {
                "image":            res.get("ma", {}).get("image_arr"),
                "original_overlay": res.get("ma", {}).get("overlay_arr"),
                "count":            res.get("ma", {}).get("count", 0),
                "adv_count":        res.get("ma", {}).get("adv_count", 0),
                "basic_count":      res.get("ma", {}).get("basic_count", 0),
                "removed_count":    res.get("ma", {}).get("removed_count", 0),
                "fp_reduction_pct": res.get("ma", {}).get("fp_reduction_pct", 0.0),
                "adv_image":        res.get("ma", {}).get("adv_image_arr"),
                "basic_image":      res.get("ma", {}).get("basic_image_arr"),
                "removed_image":    res.get("ma", {}).get("removed_image_arr"),
            } if phase == 2 else {},
            "rfnld":    {
                "image":         res.get("rfnld", {}).get("image_arr"),
                "defects_found": res.get("rfnld", {}).get("defects_found", False),
                "defect_count":  res.get("rfnld", {}).get("defect_count", 0),
            } if phase == 2 else {},
        },
        original_image = job["image_arr"],
        lang           = lang,
        report_id      = report_id,
    )

    patient_name = job["patient"].get("name", "patient").replace(" ", "_")
    filename     = f"Aakhi_Report_{patient_name}_Phase{phase}_{report_id}.pdf"

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


# ── i18n endpoint ─────────────────────────────────────────────────────────── #

@app.route("/api/i18n/<lang>")
def api_i18n(lang: str):
    allowed = {"en", "hi", "or", "bn"}
    if lang not in allowed:
        lang = "en"
    fpath = _path("static", "i18n", f"{lang}.json")
    if not os.path.exists(fpath):
        return jsonify({}), 404
    with open(fpath, encoding="utf-8") as f:
        return jsonify(json.load(f))


# ── User management ───────────────────────────────────────────────────────── #

@app.post("/api/register")
def api_register():
    data     = request.get_json() or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))
    fullname = str(data.get("full_name", username))
    if not username or not password:
        return jsonify({"ok": False, "error": "Username and password required"}), 400
    ok, msg = register_user(username, password, full_name=fullname, role="doctor")
    return jsonify({"ok": ok, "error": msg if not ok else ""})


# ── Entry point ───────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import webbrowser

    port = int(os.environ.get("PORT", 5050))
    url  = f"http://localhost:{port}"
    print(f"\n  Aakhi is running at {url}\n  Press Ctrl+C to stop.\n")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
