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
        "color_output": True, # <--- CHANGE THIS TO True
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


def set_selected(model_key: str):
    # Clear stale report when switching models
    st.session_state.pop("last_report", None)
    st.session_state.selected_model = model_key
    st.session_state.sidebar_run_click = True

def _render_report_button(user: dict, model_key: str, image_cv2: np.ndarray) -> None:
    """Render the Generate Report button and download widget."""
    report_data = st.session_state.get("last_report")
    if not report_data:
        return
    st.markdown("---")
    if st.button("📄 Generate Report", use_container_width=True, key=f"report_btn_{model_key}"):
        with st.spinner("Building PDF report ..."):
            pdf_bytes = generate_report(
                patient_name   = st.session_state.get("patient_name", "Unknown"),
                patient_age    = st.session_state.get("patient_age", 0),
                patient_gender = st.session_state.get("patient_gender", "—"),
                doctor_name    = st.session_state.get("full_name", user["username"]),
                model_title    = MODEL_INFO[model_key]["title"],
                input_image    = report_data["input"],
                output_images  = report_data["outputs"],
            )
        patient_name = st.session_state.get("patient_name", "patient").replace(" ", "_")
        filename = f"Aakhi_Report_{patient_name}_{model_key}.pdf"
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

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "MA"
    if "sidebar_run_click" not in st.session_state:
        st.session_state.sidebar_run_click = False
        
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
        
    # Add this line right here, before your Sidebar and Header code!
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

        st.markdown("---")
        st.markdown("## Models")
        if st.button("Microaneurysm (MA)"):
            set_selected("MA")
        if st.button("OD - OC Segmentation (ODOC)"):
            set_selected("ODOC")
        if st.button("DR Grading"):
            set_selected("DRG")
        if st.button("Multi-Lesion Detector (IDRiD)"):
            set_selected("LESION")

        st.markdown("---")
        st.markdown("### Quick info")
        info = MODEL_INFO[st.session_state.selected_model]
        st.write("Selected model:", info["title"])
        st.markdown(info["notes"])

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

    st.markdown(f"<div style='text-align: center;'><b>{t('active_model')}</b> {MODEL_INFO[st.session_state.selected_model]['title']}</div>", unsafe_allow_html=True)
    st.markdown("---")

    main_col, info_col = st.columns([3, 1])

    with info_col:
        info = MODEL_INFO[st.session_state.selected_model]
        st.header(info["title"])
        st.write("**Description:**")
        st.write(info["description"])
        if info["recommended_threshold"] is not None and info["recommended_batch"] is not None:
            st.write("**Recommended threshold:**", info["recommended_threshold"])
            st.write("**Recommended batch size:**", info["recommended_batch"])
        else:
            st.write("**Threshold / batch:** Not used for this model.")
        st.write("**Notes:**")
        st.write(info["notes"])

    with main_col:
        model_key = st.session_state.selected_model
        info = MODEL_INFO[model_key]

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
                    st.session_state.pop("last_report", None)  # clear any stale report
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
                st.session_state.pop("last_report", None)  # clear stale report
                st.rerun()

        st.markdown("---")

        uploaded = st.file_uploader("Upload fundus image", type=["jpg", "jpeg", "png"], key=f"uploader_{model_key}")

        if uploaded is None:
            st.info("Please upload an image to run the selected model.")
            st.session_state.sidebar_run_click = False
            return

        image_pil = Image.open(uploaded).convert("RGB")
        image_cv2 = np.array(image_pil)

        st.subheader("Input image")
        st.image(image_cv2, use_container_width=True)

        # --- DRG ---
        if model_key == "DRG":
            run_now = st.button("Run DR Grading", key=f"run_{model_key}")
            if run_now:
                if "dr_model" not in st.session_state:
                    st.session_state.dr_model = None
                if st.session_state.dr_model is None:
                    with st.spinner("Loading DR Grading model ..."):
                        try:
                            st.session_state.dr_model = load_dr_model()
                        except Exception as e:
                            st.error(f"Failed to load DR model: {e}")
                            return
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        image_pil.save(tmp.name)
                        tmp_name = tmp.name
                    with st.spinner("Predicting DR severity ..."):
                        sev = predict_dr_severity(tmp_name, st.session_state.dr_model)
                    st.success(f"Predicted DR Severity: {sev}")

                    # Build a readable severity image for the report
                    sev_img = Image.new("RGB", (600, 100), color=(255, 255, 255))
                    d = ImageDraw.Draw(sev_img)
                    try:
                        font = ImageFont.truetype("arial.ttf", 36)
                    except Exception:
                        font = ImageFont.load_default()
                    d.text((20, 28), f"DR Severity: {sev}", fill=(30, 80, 150), font=font)
                    sev_np = np.array(sev_img)

                    st.session_state["last_report"] = {
                        "input": image_cv2.copy(),
                        "outputs": [("DR Grading Result", sev_np)],
                    }
                    if os.path.exists(tmp_name):
                        os.unlink(tmp_name)
                except Exception as e:
                    st.error(f"DR Grading failed: {e}")
                    if 'tmp_name' in locals() and os.path.exists(tmp_name):
                        os.unlink(tmp_name)

            if "last_report" in st.session_state:
                _render_report_button(user, model_key, image_cv2)
            return  # DRG path ends here

        # --- MA / ODOC / LESION ---
        thr = st.number_input(
            "Threshold (probability cutoff)",
            min_value=0.0, max_value=1.0,
            value=float(info["recommended_threshold"]),
            step=0.01,
            key=f"threshold_input_{model_key}",
        )
        batch = st.number_input(
            "Batch size",
            min_value=1, max_value=256,
            value=int(info["recommended_batch"]),
            step=1,
            key=f"batch_input_{model_key}",
        )

        # Colour legend for LESION model
        if info.get("color_output") and model_key == "LESION":
            st.markdown(
                "**Colour key:** "
                "🔴 Hard Exudates &nbsp;&nbsp; 🟢 Hemorrhages &nbsp;&nbsp; "
                "🔵 Microaneurysms &nbsp;&nbsp; 🟡 Soft Exudates"
            )

        run_now = st.button("Run model on uploaded image", key=f"run_{model_key}")

        if run_now:
            if info["module"] is None or not hasattr(info["module"], "processing"):
                st.error("Processing module not available.")
                return
            processing_fn = getattr(info["module"], "processing")

            try:
                # MA: show patch-level progress bar
                if model_key == "MA":
                    progress_bar = st.progress(0, text="Starting inference...")
                    status_text  = st.empty()

                    def ma_progress(current, total):
                        pct = current / total
                        progress_bar.progress(pct, text=f"Processing patches... batch {current}/{total}")
                        status_text.caption(f"{int(pct * 100)}% complete")

                    result = processing_fn(image_cv2, float(thr), int(batch), progress_callback=ma_progress)
                    progress_bar.progress(1.0, text="Done!")
                    status_text.empty()
                    progress_bar.empty()

                # All other models: simple spinner
                else:
                    with st.spinner(f"Running {info['title']} ..."):
                        result = processing_fn(image_cv2, float(thr), int(batch))

                # -------------------------------------------------------- #
                #  Branch A: colour-coded RGB output (LESION)              #
                # -------------------------------------------------------- #
                if info.get("color_output"):
                    color_img = np.array(result, dtype=np.uint8)

                    # Side-by-side: original | segmentation
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.subheader("Input image")
                        st.image(image_cv2, use_container_width=True)
                    with col_b:
                        st.subheader("Segmentation output")
                        st.image(color_img, use_container_width=True)

                    # Blend overlay (50 % alpha)
                    blend = cv2.addWeighted(image_cv2, 0.55, color_img, 0.45, 0)
                    st.subheader("Blended overlay")
                    st.image(blend, use_container_width=True)

                    st.session_state["last_report"] = {
                        "input": image_cv2.copy(),
                        "outputs": [
                            (f"{info['title']} — Segmentation", color_img),
                            (f"{info['title']} — Overlay",      blend),
                        ],
                    }

                # -------------------------------------------------------- #
                #  Branch B: mask / probability output (MA, ODOC)           #
                # -------------------------------------------------------- #
                else:
                    if isinstance(result, (float, int)):
                        st.success(f"Model score / probability: {float(result):.4f}")
                        return

                    mask = np.array(result)
                    if mask.ndim == 3 and mask.shape[-1] == 1:
                        mask = mask[..., 0]
                    mask = mask.astype(np.float32)

                    disp    = (mask * 255.0).clip(0, 255).astype(np.uint8)
                    overlay = overlay_mask_on_rgb(image_cv2, mask > 0)

                    st.subheader("Model output (mask / probability map)")
                    st.image(disp, caption="Model output (0-255)", use_container_width=True)
                    st.subheader("Overlay (green marks)")
                    st.image(overlay, use_container_width=True)

                    output_images = [
                        ("Model Output Mask",    disp),
                        ("Overlay (green marks)", overlay),
                    ]

                    if model_key == "MA":
                        binary_mask = (mask > 0).astype(np.uint8) * 255
                        kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
                        dilated  = cv2.dilate(binary_mask, kernel, iterations=2)
                        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        circled  = image_cv2.copy()
                        for cnt in contours:
                            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                            radius = max(int(radius), 10)
                            cv2.circle(circled, (int(cx), int(cy)), radius, (255, 0, 0), 2)
                        st.subheader(f"Detected MA regions — {len(contours)} cluster(s) found")
                        st.image(circled, caption="Red circles = MA clusters", use_container_width=True)
                        output_images.append((f"MA Clusters ({len(contours)} detected)", circled))

                    st.session_state["last_report"] = {
                        "input": image_cv2.copy(),
                        "outputs": output_images,
                    }

            except Exception as e:
                st.error(f"Model inference failed: {e}")

        if "last_report" in st.session_state and model_key in ("MA", "ODOC", "LESION"):
            _render_report_button(user, model_key, image_cv2)

    st.markdown("---")
    st.caption("Tip: Place modules under ./processing and models under ./models.")


if __name__ == "__main__":
    main()