import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import sys
import base64
from helper import get_path, get_base64_image, overlay_mask_on_rgb

from auth import show_auth_screen, is_logged_in, logout, current_user
from report import generate_report
from translations import LANGUAGES, get_text

# --- set_page_config MUST be first Streamlit call ---
st.set_page_config(layout="wide", page_title="Aakhi")

# --- Initialize default language state early so functions can access it ---
if "lang_code" not in st.session_state:
    st.session_state["lang_code"] = "en"  # Default to English

# Global translation helper
def t(key):
    return get_text(st.session_state.get("lang_code", "en"), key)

try:
    import processing.processing_ma as proc_ma
except Exception as e:
    proc_ma = None
    print("Microaneurysm processing module not found:", e)

try:
    import processing.processing_odoc as proc_od
except Exception as e:
    proc_od = None
    print("OD-OC processing module not found:", e)


try:
    import processing.processing_lesion as proc_lesion
except Exception as e:
    proc_lesion = None
    print("Multi-lesion processing module not found:", e)

from processing.processing_dr_grading import load_dr_model, predict_dr_severity


MODEL_INFO = {
    "MA": {
        "title": "Microaneurysm Detector (MA)",
        "module": proc_ma,
        "description": (
            "Detects microaneurysms. Input: RGB fundus image. "
            "Output: probability map and binary mask of candidate MAs."
        ),
        "recommended_threshold": 0.9,
        "recommended_batch": 20,
        "notes": "Uses your UNet-based pipeline (processing_ma.py).",
        "color_output": False,
    },
    "ODOC": {
        "title": "Optic Disc / Optic Cup (OD-OC) Segmentation",
        "module": proc_od,
        "description": (
            "Segments optic disc and optic cup. Input: RGB fundus image. "
            "Output: segmentation masks (OD, OC) or probability maps."
        ),
        "recommended_threshold": 0.5,
        "recommended_batch": 8,
        "notes": "Module should expose processing(image, threshold, batch_size).",
        "color_output": True, 
    },
    "DRG": {
        "title": "DR Grading",
        "module": None,
        "description": (
            "Grades diabetic retinopathy severity (0–4) from a fundus image and maps it "
            "to No DR / Mild / Moderate / Severe / Proliferative DR."
        ),
        "recommended_threshold": None,
        "recommended_batch": None,
        "notes": "Loads EfficientNet-B6 weights from models/pytorch_model_effb6.bin.",
        "color_output": False,
    },
    "LESION": {
        "title": "Multi-Lesion Detector (IDRiD / FIAM)",
        "module": proc_lesion,
        "description": (
            "Detects four lesion types from a fundus image using a UNet + FIAM model "
            "trained on the IDRiD dataset. "
            "Output colour key — "
            "Red: Hard Exudates | Green: Hemorrhages | Blue: Microaneurysms | Yellow: Soft Exudates."
        ),
        "recommended_threshold": 0.5,
        "recommended_batch": 1,
        "notes": (
            "Model: Unet+FIAM_IDriD_1.2_300_cad.h5  |  "
            "Built on Python 3.11.9 / TF 2.15. "
            "Uses green channel internally. Returns a colour-coded RGB image directly."
        ),
        "color_output": True,
    },
}


def _render_report_button(user: dict, image_cv2: np.ndarray) -> None:
    """Render the Generate Report button and download widget (all models combined)."""
    report_data = st.session_state.get("last_report")
    if not report_data:
        return
    st.markdown("---")
    
    # Translated the Generate Report button fallback to 'export_btn'
    if st.button(f"📄 {t('export_btn')}", use_container_width=True, key="report_btn_all"):
        with st.spinner("Building PDF report ..."):
            pdf_bytes = generate_report(
                patient_name   = st.session_state.get("patient_name", "Unknown"),
                patient_age    = st.session_state.get("patient_age", 0),
                patient_gender = st.session_state.get("patient_gender", "—"),
                doctor_name    = st.session_state.get("full_name", user["username"]),
                model_title    = "Full Retinal Analysis (All Models)",
                input_image    = report_data["input"],
                output_images  = report_data["outputs"],
            )
        patient_name = st.session_state.get("patient_name", "patient").replace(" ", "_")
        filename = f"Aakhi_Report_{patient_name}.pdf"
        
        st.download_button(
            label=f"⬇️ Download PDF / {t('export_btn')}",
            data=pdf_bytes,
            file_name=filename,
            mime="application/pdf",
            use_container_width=True,
        )

def main():
    # ------------------------------------------------------------------ #
    #  AUTH GATE                                                           #
    # ------------------------------------------------------------------ #
    if not is_logged_in() and not st.session_state.get("guest_mode"):
        show_auth_screen()

        st.markdown("---")
        st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
        st.caption("Don't have an account?")
        if st.button("👤 Continue as Guest", use_container_width=False):
            st.session_state["guest_mode"] = True
            st.session_state["guest_user"] = {
                "username": "guest",
                "full_name": "Guest User",
            }
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()


    user = st.session_state.get("guest_user") if st.session_state.get("guest_mode") else current_user()

    # --- CUSTOM CSS FOR BUTTONS AND UI ---
    st.markdown("""
        <style>
        /* Style all standard Streamlit buttons to match the mockup */
        div.stButton > button:first-child, div.stDownloadButton > button:first-child {
            background-color: #1c4b82;
            background-image: linear-gradient(to bottom, #2a62a3, #163e6e);
            color: white;
            border: none;
            border-radius: 4px;
            padding: 10px 24px;
            font-weight: bold;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.2);
            width: 100%;
        }
        /* Hover effect for buttons */
        div.stButton > button:first-child:hover, div.stDownloadButton > button:first-child:hover {
            background-image: linear-gradient(to bottom, #3474bc, #1c4b82);
            border-color: #1c4b82;
            color: white;
        }
        /* Make the file uploader look cleaner */
        section[data-testid="stFileUploadDropzone"] {
            background-color: #f0f4f8;
            border: 2px dashed #2a62a3;
        }
        </style>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------ #
    #  SIDEBAR                                                             #
    # ------------------------------------------------------------------ #
    with st.sidebar:
        # --- Language Selector ---
        selected_lang_name = st.selectbox(
            "🌐 Select Language / भाषा चुनें", 
            options=list(LANGUAGES.keys()),
            index=list(LANGUAGES.values()).index(st.session_state["lang_code"])
        )
        
        # Update session state if changed
        new_lang_code = LANGUAGES[selected_lang_name]
        if new_lang_code != st.session_state["lang_code"]:
            st.session_state["lang_code"] = new_lang_code
            st.rerun()
            
        st.markdown("---")
        if st.session_state.get("guest_mode"):
            st.markdown("👤 **Guest User**")
            st.caption("*Browsing as guest*")
            if st.button("Exit Guest / Login", use_container_width=True):
                st.session_state.pop("guest_mode", None)
                st.session_state.pop("guest_user", None)
                st.rerun()
        else:
            st.markdown(f"👤 **{user['full_name']}**")
            st.caption(f"@{user['username']}")
            if st.button("Logout", use_container_width=True):
                logout()
                st.rerun()



    # ------------------------------------------------------------------ #
    #  HEADER                                                              #
    # ------------------------------------------------------------------ #
    logo_path = get_path("aakhi_logo.png") # Make sure this filename is correct
    
    if os.path.exists(logo_path):
        img_base64 = get_base64_image(logo_path)
        
        # This HTML puts the logo right next to the "AAKHI" text and centers everything perfectly
        st.markdown(f"""
            <div style='text-align: center; padding-bottom: 10px;'>
                <div style='display: inline-flex; align-items: center; justify-content: center;'>
                    <img src='data:image/png;base64,{img_base64}' style='width: 120px; margin-right: 15px;'>
                    <h1 style='color: #1c4b82; font-size: 3.5em; margin: 0; font-weight: bold;'>{t("app_title")}</h1>
                </div>
                <h3 style='color: #333; margin-top: 10px; font-weight: 600;'>{t("app_subtitle")}</h3>
                <h4 style='color: #555; font-style: italic; margin-top: 5px;'>{t("app_motto")}</h4>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback if logo is missing
        st.markdown(f"""
            <div style='text-align: center; padding-bottom: 10px;'>
                <h1 style='color: #1c4b82; font-size: 3.5em; margin-bottom: 0; font-weight: bold;'>{t("app_title")}</h1>
                <h3 style='color: #333; margin-top: 5px; font-weight: 600;'>{t("app_subtitle")}</h3>
                <h4 style='color: #555; font-style: italic; margin-top: 5px;'>{t("app_motto")}</h4>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    main_col = st.container()

    with main_col:
        # ------------------------------------------------------------------ #
        #  PATIENT INFO                                                        #
        # ------------------------------------------------------------------ #
        if "patient_confirmed" not in st.session_state:
            st.session_state["patient_confirmed"] = False

        if not st.session_state["patient_confirmed"]:
            # Translated Title & Prompts
            st.subheader(f"{t('patient_details')}")
            st.caption("Please enter patient details before uploading an image.")
            
            p_name   = st.text_input(t("patient_name"), placeholder="e.g. Rachit Jain", key="input_patient_name")
            p_age    = st.number_input(t("patient_age"), min_value=1, max_value=120, value=22, step=1, key="input_patient_age")
            p_gender = st.selectbox(t("gender"), ["Male", "Female", "Other"], key="input_patient_gender")
            
            if st.button(t("confirm"), use_container_width=True):
                if not p_name.strip():
                    st.error("Please enter the patient's name.")
                else:
                    st.session_state["patient_name"]    = p_name.strip()
                    st.session_state["patient_age"]     = int(p_age)
                    st.session_state["patient_gender"]  = p_gender
                    st.session_state["patient_confirmed"] = True
                    st.session_state.pop("last_report", None)
                    st.rerun()
            return

        # Patient banner
        p_name   = st.session_state["patient_name"]
        p_age    = st.session_state["patient_age"]
        p_gender = st.session_state["patient_gender"]
        banner_col, change_col = st.columns([4, 1])
        with banner_col:
            st.success(f"👤 Patient: **{p_name}** | {t('patient_age')}: **{p_age}** | {t('gender')}: **{p_gender}**")
        with change_col:
            if st.button(t("change_patient"), use_container_width=True):
                st.session_state["patient_confirmed"] = False
                st.session_state["patient_name"]      = ""
                st.session_state["patient_age"]       = 0
                st.session_state["patient_gender"]    = ""
                st.session_state.pop("last_report", None)
                st.rerun()

        st.markdown("---")

        # Translated File Upload 
        uploaded = st.file_uploader(t("upload_prompt"), type=["jpg", "jpeg", "png"], key="uploader_main")

        if uploaded is None:
            st.info("Please upload a fundus image to begin analysis.")
            return

        image_pil = Image.open(uploaded).convert("RGB")
        image_cv2 = np.array(image_pil)

        st.subheader("Input Image")
        st.image(image_cv2, use_container_width=True)

        # Translated Analyze button
        analyze_btn = st.button(t("analyze_btn"), use_container_width=True)

        if analyze_btn:
            all_outputs = []  # list of (label, image_np) for the report
            
            # ------------------------------------------------------------------ #
            #  OD-OC SEGMENTATION & CLASSIFICATION                                #
            # ------------------------------------------------------------------ #
            st.markdown("---")
            st.subheader("Optic Disc / Cup (OD-OC) & Glaucoma Classification")
            info_od = MODEL_INFO["ODOC"]
            if info_od["module"] is None or not hasattr(info_od["module"], "processing"):
                st.warning("ODOC processing module not available.")
            else:
                try:
                    thr_od   = float(info_od["recommended_threshold"])
                    batch_od = int(info_od["recommended_batch"])
                    
                    with st.spinner("Running OD-OC segmentation & classifying features..."):
                        outlined_img, features, rf_pred, rf_prob = info_od["module"].processing(image_cv2, thr_od, batch_od)

                    st.image(outlined_img, caption="OD-OC Outlines (Green: Disc, Red: Cup)", use_container_width=True)

                    if features is None:
                        st.warning("Could not clearly detect Optic Disc in this image. Classification skipped.")
                    else:
                        cdr_value = features.get('v_cdr', 0.0)
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Vertical Cup-to-Disc Ratio (vCDR)", f"{cdr_value:.3f}")
                        col2.metric("Glaucoma Probability (RF Model)", f"{rf_prob * 100:.1f}%")
                        
                        if rf_pred == "Glaucoma":
                            st.error(f"⚠️ High Risk: AAKHI predicts Glaucoma.")
                        else:
                            st.success(f"✅ Low Risk: AAKHI predicts Normal.")

                        # --- Build OD-OC Statistics Image for PDF Report ---
                        odoc_stats_img = Image.new("RGB", (900, 180), color=(255, 255, 255))
                        d = ImageDraw.Draw(odoc_stats_img)
                            
                        d.text((20, 20), "OD-OC Segmentation & RF Classification", fill=(30, 80, 150))
                        d.text((60, 65), f"Calculated vCDR: {cdr_value:.3f}", fill=(50, 50, 50))
                        d.text((60, 95), f"Model Probability: {rf_prob * 100:.1f}%", fill=(50, 50, 50))
                        
                        status_color = (255, 0, 0) if rf_pred == "Glaucoma" else (0, 150, 0)
                        d.text((60, 125), f"Prediction: {rf_pred.upper()}", fill=status_color)
                        # ---------------------------------------------------

                        all_outputs.extend([
                            (f"OD-OC Analysis (vCDR: {cdr_value:.3f} | Pred: {rf_pred})", outlined_img),
                            ("OD-OC & Glaucoma — Report Summary", np.array(odoc_stats_img)),
                        ])
                except Exception as e:
                    st.error(f"OD-OC inference failed: {e}")
                
            # ------------------------------------------------------------------ #
            #  MICROANEURYSM (MA) — Advanced & Basic Modes                        #
            # ------------------------------------------------------------------ #
            st.markdown("---")
            st.subheader("Microaneurysm Detector (MA)")
            info_ma = MODEL_INFO["MA"]
            if info_ma["module"] is None or not hasattr(info_ma["module"], "processing"):
                st.warning("MA processing module not available.")
            else:
                try:
                    thr_ma   = float(info_ma["recommended_threshold"])
                    batch_ma = int(info_ma["recommended_batch"])
                    progress_bar = st.progress(0, text="Starting MA inference...")
                    status_text  = st.empty()

                    def ma_progress(current, total):
                        pct = current / total
                        progress_bar.progress(pct, text=f"Processing patches... batch {current}/{total}")
                        status_text.caption(f"{int(pct * 100)}% complete")

                    result_ma = info_ma["module"].processing(image_cv2, thr_ma, batch_ma, progress_callback=ma_progress)
                    progress_bar.progress(1.0, text="Done!")
                    status_text.empty()
                    progress_bar.empty()

                    # --- Unpack the dict returned by the updated processing_ma ---
                    prob_map       = result_ma["prob_map"]
                    baseline_mask  = result_ma["baseline_mask"]
                    refined_mask   = result_ma["refined_mask"]
                    adv_count      = result_ma["baseline_count"]
                    basic_count    = result_ma["refined_count"]
                    removed_count  = result_ma["removed_count"]
                    removed_locs   = result_ma["removed_locations"]

                    # --- Probability map visualisation (shared) ---
                    disp = (prob_map * 255.0).clip(0, 255).astype(np.uint8)

                    # ============================================================ #
                    #  Helper: build circled-cluster image from a binary mask       #
                    # ============================================================ #
                    def _build_circled_image(base_img, bin_mask, label_prefix=""):
                        """Draw numbered circles around clusters in bin_mask and return (image, ma_data, ma_stats)."""
                        bm255 = (bin_mask > 0).astype(np.uint8) * 255
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
                        dilated = cv2.dilate(bm255, kernel, iterations=2)
                        cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        out = base_img.copy()

                        data = []
                        for idx, cnt in enumerate(cnts):
                            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                            data.append({
                                "id": idx + 1,
                                "cx": int(cx), "cy": int(cy),
                                "radius": radius,
                                "display_radius": max(int(radius), 10),
                            })

                        top5 = [m["id"] for m in sorted(data, key=lambda x: x["radius"], reverse=True)[:5]]
                        stats = []
                        for m in data:
                            color = (255, 0, 0) if m["id"] in top5 else (255, 255, 0)
                            cv2.circle(out, (m["cx"], m["cy"]), m["display_radius"], color, 2)
                            cv2.putText(out, str(m["id"]),
                                        (m["cx"] + m["display_radius"], m["cy"] - m["display_radius"]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                            stats.append((m["id"], round(m["radius"], 2)))

                        return out, len(cnts), data, stats, top5

                    # ============================================================ #
                    #  Helper: build summary stats image for PDF report            #
                    # ============================================================ #
                    def _build_stats_image(title_text, num, ma_stats_list, top5_list):
                        rows_needed = (num // 5) + 1
                        bg_h = max(100, 60 + rows_needed * 30)
                        img = Image.new("RGB", (900, bg_h), color=(255, 255, 255))
                        d = ImageDraw.Draw(img)
                        try:
                            ft = ImageFont.truetype("arial.ttf", 22)
                            ft2 = ImageFont.truetype("arial.ttf", 16)
                        except Exception:
                            ft = ImageFont.load_default()
                            ft2 = ImageFont.load_default()
                        d.text((20, 15), title_text, fill=(30, 80, 150), font=ft)
                        xo, yo = 55, 55
                        for mid, rad in ma_stats_list:
                            tc = (255, 0, 0) if mid in top5_list else (50, 50, 50)
                            d.text((xo, yo), f"MA #{mid}: {rad}px", fill=tc, font=ft2)
                            xo += 160
                            if xo > 800:
                                xo = 55
                                yo += 30
                        return np.array(img)

                    # ============================================================ #
                    #  COMPARISON SUMMARY (shown first)                            #
                    # ============================================================ #
                    reduction_pct = (100.0 * removed_count / max(adv_count, 1))
                    st.markdown("#### 📊 MA Detection Comparison")
                    cmp_cols = st.columns(3)
                    cmp_cols[0].metric("🔬 Maximum (All Candidates)", f"{adv_count} MA(s)")
                    cmp_cols[1].metric("🎯 Minimum (FP-Reduced)", f"{basic_count} MA(s)")
                    cmp_cols[2].metric("🗑️ False Positives Removed", f"{removed_count}", delta=f"−{reduction_pct:.1f}%", delta_color="normal")

                    # --- Shared probability map row ---
                    st.markdown("##### Probability Map")
                    col_pm1, col_pm2 = st.columns(2)
                    with col_pm1:
                        st.image(disp, caption="Probability map (0–255)", use_container_width=True)
                    with col_pm2:
                        overlay_adv = overlay_mask_on_rgb(image_cv2, baseline_mask > 0)
                        st.image(overlay_adv, caption="Green overlay (Maximum — all candidates)", use_container_width=True)

                    # ============================================================ #
                    #  MAXIMUM MODE (all MA candidates)                            #
                    # ============================================================ #
                    st.markdown("---")
                    st.markdown("#### 🔬 Maximum Mode — All MA Candidates (Higher Sensitivity)")
                    st.caption("Shows every candidate suspected by the model, including potential false positives.")

                    circled_adv, n_adv, _, stats_adv, top5_adv = _build_circled_image(image_cv2, baseline_mask)
                    st.image(circled_adv,
                             caption=f"Maximum — {n_adv} MA cluster(s) suspected (to be clinically correlated by a professional) (Top 5 largest in Red)",
                             use_container_width=True)

                    # ============================================================ #
                    #  MINIMUM MODE (FP-reduced)                                     #
                    # ============================================================ #
                    st.markdown("---")
                    st.markdown("#### 🎯 Minimum Mode — Refined MA Candidates (Fewer False Positives)")
                    st.caption("False positives removed using area, circularity, and confidence filters.")

                    circled_basic, n_basic, _, stats_basic, top5_basic = _build_circled_image(image_cv2, refined_mask)

                    # Also draw markers for removed FP locations
                    removed_vis = image_cv2.copy()
                    for (rx, ry) in removed_locs:
                        cv2.circle(removed_vis, (rx, ry), 12, (255, 0, 0), 2)
                        cv2.putText(removed_vis, "FP", (rx + 10, ry - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        st.image(circled_basic,
                                 caption=f"Minimum — {n_basic} MA cluster(s) kept (Top 5 largest in Red)",
                                 use_container_width=True)
                    with col_b2:
                        st.image(removed_vis,
                                 caption=f"Removed false positives — {removed_count} candidate(s) filtered out",
                                 use_container_width=True)

                    # --- Build summary images for the PDF report ---
                    stats_img_adv = _build_stats_image(
                        f"Maximum Mode — {n_adv} MAs Suspected (to be clinically correlated by a professional) (Top 5 in Red)", n_adv, stats_adv, top5_adv)
                    stats_img_basic = _build_stats_image(
                        f"Minimum Mode — {n_basic} MAs Kept (Top 5 in Red)", n_basic, stats_basic, top5_basic)

                    # --- Build comparison summary image for report ---
                    cmp_img = Image.new("RGB", (900, 120), color=(255, 255, 255))
                    d_cmp = ImageDraw.Draw(cmp_img)
                    try:
                        f_t = ImageFont.truetype("arial.ttf", 22)
                        f_b = ImageFont.truetype("arial.ttf", 18)
                    except Exception:
                        f_t = ImageFont.load_default()
                        f_b = ImageFont.load_default()
                    d_cmp.text((20, 15), "MA Detection Comparison", fill=(30, 80, 150), font=f_t)
                    d_cmp.text((40, 55), f"Maximum (All): {n_adv} MA(s)", fill=(50, 50, 50), font=f_b)
                    d_cmp.text((340, 55), f"Minimum (Refined): {n_basic} MA(s)", fill=(0, 128, 0), font=f_b)
                    d_cmp.text((640, 55), f"Removed: {removed_count} ({reduction_pct:.1f}%)", fill=(200, 0, 0), font=f_b)

                    all_outputs.extend([
                        ("MA — Probability Map", disp),
                        ("MA — Maximum Mode — Overlay (All Candidates)", overlay_adv),
                        (f"MA — Maximum Mode — Clusters ({n_adv} suspected (to be clinically correlated by a professional))", circled_adv),
                        ("MA — Maximum Mode — Detection Summary", stats_img_adv),
                        (f"MA — Minimum Mode — Clusters ({n_basic} kept, {removed_count} removed)", circled_basic),
                        (f"MA — Minimum Mode — Removed False Positives ({removed_count})", removed_vis),
                        ("MA — Minimum Mode — Detection Summary", stats_img_basic),
                        ("MA — Comparison Summary", np.array(cmp_img)),
                    ])
                except Exception as e:
                    st.error(f"MA inference failed: {e}")
                    
            # ------------------------------------------------------------------ #
            #  MULTI-LESION DETECTOR                                              #
            # ------------------------------------------------------------------ #
            st.markdown("---")
            st.subheader("Multi-Lesion Detector")
            info_lesion = MODEL_INFO["LESION"]
            if info_lesion["module"] is None or not hasattr(info_lesion["module"], "processing"):
                st.warning("Multi-Lesion processing module not available.")
            else:
                try:
                    thr_l   = float(info_lesion["recommended_threshold"])
                    batch_l = int(info_lesion["recommended_batch"])
                    with st.spinner("Running Multi-Lesion detection ..."):
                        result_lesion = info_lesion["module"].processing(image_cv2, thr_l, batch_l)
                    color_img_l = np.array(result_lesion, dtype=np.uint8)
                    blend_l = cv2.addWeighted(image_cv2, 0.55, color_img_l, 0.45, 0)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.image(color_img_l, caption="Lesion segmentation", use_container_width=True)
                    with col_b:
                        st.image(blend_l, caption="Blended overlay", use_container_width=True)

                    # --- 1. Calculate Lesion Statistics ---
                    lesion_colors = {
                        "Hard Exudates": (255, 0, 0),     # Red
                        "Hemorrhages": (0, 255, 0),       # Green
                        "Microaneurysms": (0, 0, 255),    # Blue
                        "Soft Exudates": (255, 255, 0)    # Yellow
                    }
                    
                    # Calculate total pixels in the image (Height x Width)
                    total_image_pixels = color_img_l.shape[0] * color_img_l.shape[1]
                    
                    stats = {}
                    for name, color in lesion_colors.items():
                        # Create a binary mask for this specific color channel
                        mask = np.all(color_img_l == color, axis=-1).astype(np.uint8) * 255
                        area_px = int(np.count_nonzero(mask))
                        
                        # Calculate Area Percentage
                        area_pct = (area_px / total_image_pixels) * 100 if total_image_pixels > 0 else 0.0
                        
                        # Find contours to count discrete clusters
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        count = len(contours)
                        
                        stats[name] = {"count": count, "area_pct": area_pct, "color": color}

                    # --- Show Metrics in the Streamlit UI ---
                    st.markdown("**Detected Lesion Statistics & Colour Key:**")
                    
                    # Map lesion types to colored circle emojis for the UI legend
                    ui_legend = {
                        "Hard Exudates": "🔴",     # Red
                        "Hemorrhages": "🟢",       # Green
                        "Microaneurysms": "🔵",    # Blue
                        "Soft Exudates": "🟡"      # Yellow
                    }
                    
                    m_cols = st.columns(4)
                    for idx, (name, data) in enumerate(stats.items()):
                        # Combine the emoji and the name for the metric label
                        display_name = f"{ui_legend.get(name, '')} {name}"
                        m_cols[idx].metric(
                            label=display_name, 
                            value=f"{data['count']} clusters", 
                            delta=f"{data['area_pct']:.4f}% area", 
                            delta_color="off"
                        )
                        
                    # --- 2. Build Statistics Table Image for the PDF Report ---
                    stats_img = Image.new("RGB", (900, 240), color=(255, 255, 255))
                    d = ImageDraw.Draw(stats_img)
                    try:
                        font_title = ImageFont.truetype("arial.ttf", 22)
                        font_header = ImageFont.truetype("arial.ttf", 18)
                        font_text = ImageFont.truetype("arial.ttf", 16)
                    except Exception:
                        font_title = ImageFont.load_default()
                        font_header = ImageFont.load_default()
                        font_text = ImageFont.load_default()
                        
                    d.text((20, 20), "Multi-Lesion Detection Statistics & Legend", fill=(30, 80, 150), font=font_title)
                    
                    # Table Headers
                    y_offset = 65
                    d.text((60, y_offset), "Lesion Type", fill=(50, 50, 50), font=font_header)
                    d.text((350, y_offset), "Cluster Count", fill=(50, 50, 50), font=font_header)
                    d.text((550, y_offset), "Total Area (%)", fill=(50, 50, 50), font=font_header)
                    
                    y_offset += 30
                    d.line([(20, y_offset), (880, y_offset)], fill=(200, 200, 200), width=2)
                    y_offset += 15

                    # Table Rows
                    for name, data in stats.items():
                        # Draw a small color box as the visual legend
                        d.rectangle([20, y_offset + 2, 40, y_offset + 18], fill=data["color"], outline=(100, 100, 100))
                        
                        # Write the data for the row
                        d.text((60, y_offset), name, fill=(50, 50, 50), font=font_text)
                        d.text((350, y_offset), str(data["count"]), fill=(50, 50, 50), font=font_text)
                        d.text((550, y_offset), f"{data['area_pct']:.4f}%", fill=(50, 50, 50), font=font_text)
                        
                        y_offset += 30
                    # ----------------------------------------------------------

                    all_outputs.extend([
                        ("Lesion — Statistics Table", np.array(stats_img)),
                        ("Lesion — Segmentation", color_img_l),
                        ("Lesion — Overlay", blend_l),
                    ])
                except Exception as e:
                    st.error(f"Multi-Lesion inference failed: {e}")

            # ------------------------------------------------------------------ #
            #  DR GRADING                                                         #
            # ------------------------------------------------------------------ #
            st.markdown("---")
            st.subheader("DR Grading")
            try:
                if "dr_model" not in st.session_state or st.session_state.dr_model is None:
                    with st.spinner("Loading DR Grading model ..."):
                        st.session_state.dr_model = load_dr_model()
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    image_pil.save(tmp.name)
                    tmp_name = tmp.name
                with st.spinner("Predicting DR severity ..."):
                    sev = predict_dr_severity(tmp_name, st.session_state.dr_model)
                st.success(f"Predicted DR Severity: **{sev}**")
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)

                sev_img = Image.new("RGB", (600, 100), color=(255, 255, 255))
                d = ImageDraw.Draw(sev_img)
                try:
                    font = ImageFont.truetype("arial.ttf", 36)
                except Exception:
                    font = ImageFont.load_default()
                d.text((20, 28), f"DR Severity: {sev}", fill=(30, 80, 150), font=font)
                all_outputs.append((f"DR Grading Result: {sev}", None))
            except Exception as e:
                st.error(f"DR Grading failed: {e}")
            
            # Save combined report data
            st.session_state["last_report"] = {
                "input": image_cv2.copy(),
                "outputs": all_outputs,
            }

        if "last_report" in st.session_state:
            _render_report_button(user, image_cv2)

    st.markdown("---")


if __name__ == "__main__":
    main()