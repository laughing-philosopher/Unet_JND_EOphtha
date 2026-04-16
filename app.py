import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import sys
import base64

from auth import show_auth_screen, is_logged_in, logout, current_user
from report import generate_report
from translations import LANGUAGES, get_text

# --- set_page_config MUST be first Streamlit call ---
st.set_page_config(layout="wide", page_title="Aakhi")

# --- Helper for PyInstaller bundling ---
def get_path(filename):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return filename

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

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


def overlay_mask_on_rgb(rgb_img, mask, alpha=0.6):
    out = rgb_img.copy()
    mask_bool = (mask > 0)
    out[mask_bool, 1] = 255
    return out


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
    if st.button("📄 Generate Report", use_container_width=True, key="report_btn_all"):
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
        filename = f"Aakhi_Report_{patient_name}_AllModels.pdf"
        st.download_button(
            label="⬇️ Download Report PDF",
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


    if "lang_code" not in st.session_state:
        st.session_state["lang_code"] = "en"  # Default to English

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
    
    if "lang_code" not in st.session_state:
        st.session_state["lang_code"] = "en"  # Default to English
        
    t = lambda key: get_text(st.session_state["lang_code"], key)

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
            st.subheader("🧑‍⚕️ Patient Details")
            st.caption("Please enter patient details before uploading an image.")
            p_name   = st.text_input("Patient Name", placeholder="e.g. Rachit Jain", key="input_patient_name")
            p_age    = st.number_input("Patient Age", min_value=1, max_value=120, value=22, step=1, key="input_patient_age")
            p_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="input_patient_gender")
            if st.button("Confirm Patient", use_container_width=True):
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
            st.success(f"👤 Patient: **{p_name}** | Age: **{p_age}** | Gender: **{p_gender}**")
        with change_col:
            if st.button("Change Patient", use_container_width=True):
                st.session_state["patient_confirmed"] = False
                st.session_state["patient_name"]      = ""
                st.session_state["patient_age"]       = 0
                st.session_state["patient_gender"]    = ""
                st.session_state.pop("last_report", None)
                st.rerun()

        st.markdown("---")

        uploaded = st.file_uploader("Upload fundus image", type=["jpg", "jpeg", "png"], key="uploader_main")

        if uploaded is None:
            st.info("Please upload a fundus image to begin analysis.")
            return

        image_pil = Image.open(uploaded).convert("RGB")
        image_cv2 = np.array(image_pil)

        st.subheader("Input Image")
        st.image(image_cv2, use_container_width=True)

        analyze_btn = st.button("🔍 Analyze", use_container_width=True)

        if analyze_btn:
            all_outputs = []  # list of (label, image_np) for the report

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
                all_outputs.append(("DR Grading Result", np.array(sev_img)))
            except Exception as e:
                st.error(f"DR Grading failed: {e}")

            # ------------------------------------------------------------------ #
            #  MICROANEURYSM (MA)                                                 #
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

                    mask = np.array(result_ma)
                    if mask.ndim == 3 and mask.shape[-1] == 1:
                        mask = mask[..., 0]
                    mask = mask.astype(np.float32)
                    disp    = (mask * 255.0).clip(0, 255).astype(np.uint8)
                    overlay = overlay_mask_on_rgb(image_cv2, mask > 0)

                    binary_mask = (mask > 0).astype(np.uint8) * 255
                    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
                    dilated  = cv2.dilate(binary_mask, kernel, iterations=2)
                    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    circled  = image_cv2.copy()
                    for cnt in contours:
                        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                        radius = max(int(radius), 10)
                        cv2.circle(circled, (int(cx), int(cy)), radius, (255, 0, 0), 2)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.image(disp, caption="Probability map (0–255)", use_container_width=True)
                    with col_b:
                        st.image(overlay, caption="Green overlay", use_container_width=True)
                    st.image(circled, caption=f"MA clusters — {len(contours)} cluster(s) found (red circles)", use_container_width=True)

                    all_outputs.extend([
                        ("MA — Probability Map", disp),
                        ("MA — Overlay", overlay),
                        (f"MA — Clusters ({len(contours)} detected)", circled),
                    ])
                except Exception as e:
                    st.error(f"MA inference failed: {e}")

            # ------------------------------------------------------------------ #
            #  OD-OC SEGMENTATION                                                 #
            # ------------------------------------------------------------------ #
            st.markdown("---")
            st.subheader("Optic Disc / Optic Cup (OD-OC) Segmentation")
            info_od = MODEL_INFO["ODOC"]
            if info_od["module"] is None or not hasattr(info_od["module"], "processing"):
                st.warning("ODOC processing module not available.")
            else:
                try:
                    thr_od   = float(info_od["recommended_threshold"])
                    batch_od = int(info_od["recommended_batch"])
                    with st.spinner("Running OD-OC segmentation ..."):
                        # Unpack the newly added height variables
                        color_img_od, cdr_value, od_height, oc_height = info_od["module"].processing(image_cv2, thr_od, batch_od)
                    blend_od = cv2.addWeighted(image_cv2, 0.55, color_img_od, 0.45, 0)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.image(color_img_od, caption="OD-OC Segmentation", use_container_width=True)
                    with col_b:
                        st.image(blend_od, caption="Blended overlay", use_container_width=True)

                    st.metric("Vertical Cup-to-Disc Ratio (vCDR)", f"{cdr_value:.3f}")
                    st.caption(f"Calculated using Optic Disc vertical diameter: **{od_height}px**, Optic Cup vertical diameter: **{oc_height}px**")
                    
                    if cdr_value > 0.65:
                        st.error(f"⚠️ CDR is {cdr_value:.3f} — possible signs of Glaucoma. Please consult an ophthalmologist.")
                    else:
                        st.success(f"CDR is {cdr_value:.3f} — within normal range.")

                    # Append the heights to the label so it prints natively in the PDF report
                    all_outputs.extend([
                        ("OD-OC — Segmentation", color_img_od),
                        ("OD-OC — Overlay", blend_od),
                        (f"OD-OC — CDR: {cdr_value:.3f} (OD Height: {od_height}px | OC Height: {oc_height}px)", color_img_od),
                    ])
                except Exception as e:
                    st.error(f"OD-OC inference failed: {e}")

           # ------------------------------------------------------------------ #
            #  MULTI-LESION DETECTOR                                              #
            # ------------------------------------------------------------------ #
            st.markdown("---")
            st.subheader("Multi-Lesion Detector (IDRiD / FIAM)")
            st.markdown(
                "**Colour key:** "
                "🔴 Hard Exudates &nbsp;&nbsp; 🟢 Hemorrhages &nbsp;&nbsp; "
                "🔵 Microaneurysms &nbsp;&nbsp; 🟡 Soft Exudates"
            )
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

                    # --- Build a legend image explicitly for the PDF report ---
                    legend_img = Image.new("RGB", (900, 80), color=(255, 255, 255))
                    d = ImageDraw.Draw(legend_img)
                    try:
                        font = ImageFont.truetype("arial.ttf", 22)
                    except Exception:
                        font = ImageFont.load_default()
                    
                    legend_text = "Colour Key:  Red=Hard Exudates | Green=Hemorrhages | Blue=Microaneurysms | Yellow=Soft Exudates"
                    d.text((20, 25), legend_text, fill=(30, 80, 150), font=font)
                    # ----------------------------------------------------------

                    all_outputs.extend([
                        ("Lesion — Colour Key", np.array(legend_img)),
                        ("Lesion — Segmentation", color_img_l),
                        ("Lesion — Overlay", blend_l),
                    ])
                except Exception as e:
                    st.error(f"Multi-Lesion inference failed: {e}")

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