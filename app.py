import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import os
import sys

from auth import show_auth_screen, is_logged_in, logout, current_user

# --- Helper for PyInstaller bundling ---
def get_path(filename):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return filename

# --- Optional: coordinate picker for RFNLD ---
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_COORD_PICKER = True
except Exception:
    HAS_COORD_PICKER = False

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
    import processing.processing_rnfld as proc_rfnld
except Exception as e:
    proc_rfnld = None
    print("RFNLD processing module not found:", e)

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
    },
    "RFNLD": {
        "title": "RFNLD Detector",
        "module": proc_rfnld,
        "description": (
            "Detects RNFL defects along a ring around optic disc. "
            "Requires user to mark two clicks: center (C) and a rim point (R). "
            "Output: original image with detected line(s) overlayed."
        ),
        "recommended_threshold": None,
        "recommended_batch": None,
        "notes": (
            "This model does not use threshold/batch. It needs two clicks to define the ROI: "
            "first click = disc center (C), second click = rim point (R)."
        ),
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
    },
}


def set_selected(model_key: str):
    st.session_state.selected_model = model_key
    st.session_state.sidebar_run_click = True


def main():
    # ------------------------------------------------------------------ #
    #  AUTH GATE — show login screen if not logged in, then stop          #
    # ------------------------------------------------------------------ #
    if not is_logged_in():
        show_auth_screen()
        st.stop()   # <-- nothing below renders until logged in

    # From here on, user is authenticated
    st.set_page_config(layout="wide", page_title="Aakhi")

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "MA"
    if "sidebar_run_click" not in st.session_state:
        st.session_state.sidebar_run_click = False

    user = current_user()

    # ------------------------------------------------------------------ #
    #  SIDEBAR                                                             #
    # ------------------------------------------------------------------ #
    with st.sidebar:
        # Logged-in user info + logout
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
        if st.button("RFNLD Detector (RFNLD)"):
            set_selected("RFNLD")
        if st.button("DR Grading"):
            set_selected("DRG")

        st.markdown("---")
        st.markdown("### Quick info")
        info = MODEL_INFO[st.session_state.selected_model]
        st.write("Selected model:", info["title"])
        st.markdown(info["notes"])

    # ------------------------------------------------------------------ #
    #  HEADER                                                              #
    # ------------------------------------------------------------------ #
    col_left, col_right = st.columns([3, 1])
    with col_left:
        st.title("Aakhi")
        st.subheader("Retinal Image Analysis")
        st.markdown(f"**Active model:** {MODEL_INFO[st.session_state.selected_model]['title']}")
    with col_right:
        logo_path = get_path("iitbbs logo.png")
        if os.path.exists(logo_path):
            st.image(Image.open(logo_path), width=200)

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

        uploaded = st.file_uploader("Upload fundus image", type=["jpg", "jpeg", "png"], key=f"uploader_{model_key}")

        if uploaded is None:
            st.info("Please upload an image to run the selected model.")
            st.session_state.sidebar_run_click = False
            return

        image_pil = Image.open(uploaded).convert("RGB")
        image_cv2 = np.array(image_pil)

        st.subheader("Input image")
        st.image(image_cv2, use_column_width=True)

        # --- RFNLD ---
        if model_key == "RFNLD":
            if not HAS_COORD_PICKER:
                st.error("streamlit-image-coordinates is required. pip install streamlit-image-coordinates")
                return

            st.markdown("**Step 1:** Click the *disc center (C)* on the image below.")
            click_c = streamlit_image_coordinates(image_pil, key="click_center")

            if click_c:
                cx, cy = click_c["x"], click_c["y"]
                img_mark1 = image_pil.copy()
                d = ImageDraw.Draw(img_mark1)
                r = 3
                d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(0, 255, 0), width=2)

                st.markdown("**Step 2:** Click a *rim point (R)* to define the radius.")
                click_r = streamlit_image_coordinates(img_mark1, key="click_rim")

                if click_r:
                    rx, ry = click_r["x"], click_r["y"]
                    run_now = st.button("Run RFNLD on ROI")

                    if run_now:
                        if info["module"] is None or not hasattr(info["module"], "processing"):
                            st.error("RFNLD processing module not available.")
                            return
                        processing_fn = getattr(info["module"], "processing")
                        coord_center = {"x": int(cx), "y": int(cy)}
                        coord_rim = {"x": int(rx), "y": int(ry)}

                        with st.spinner("Running RFNLD Detector ..."):
                            try:
                                out_img = processing_fn(image_cv2, coord_rim, coord_center)
                                if out_img is None:
                                    st.error("Model returned None.")
                                    return
                                out_np = np.array(out_img)
                                if out_np.ndim == 3 and out_np.shape[2] == 3:
                                    out_np = cv2.cvtColor(out_np, cv2.COLOR_BGR2RGB)
                                st.subheader("RFNLD output")
                                st.image(out_np, use_column_width=True)
                            except Exception as e:
                                st.error(f"Model inference failed: {e}")
            return

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
                    if os.path.exists(tmp_name):
                        os.unlink(tmp_name)
                except Exception as e:
                    st.error(f"DR Grading failed: {e}")
                    if 'tmp_name' in locals() and os.path.exists(tmp_name):
                        os.unlink(tmp_name)
            return

        # --- MA / ODOC ---
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

        run_now = st.button("Run model on uploaded image", key=f"run_{model_key}")

        if run_now:
            if info["module"] is None or not hasattr(info["module"], "processing"):
                st.error("Processing module not available.")
                return
            processing_fn = getattr(info["module"], "processing")

            with st.spinner(f"Running {info['title']} ..."):
                try:
                    result = processing_fn(image_cv2, float(thr), int(batch))

                    if isinstance(result, (float, int)):
                        st.success(f"Model score / probability: {float(result):.4f}")
                        return

                    mask = np.array(result)
                    if mask.ndim == 3 and mask.shape[-1] == 1:
                        mask = mask[..., 0]
                    mask = mask.astype(np.float32)

                    st.subheader("Model output (mask / probability map)")
                    disp = (mask * 255.0).clip(0, 255).astype(np.uint8)
                    st.image(disp, caption="Model output (0-255)", use_column_width=True)

                    overlay = overlay_mask_on_rgb(image_cv2, mask > 0)
                    st.subheader("Overlay (green marks)")
                    st.image(overlay, use_column_width=True)

                    if model_key == "MA":
                        binary_mask = (mask > 0).astype(np.uint8) * 255
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
                        dilated = cv2.dilate(binary_mask, kernel, iterations=2)
                        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        circled = image_cv2.copy()
                        for cnt in contours:
                            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                            radius = max(int(radius), 10)
                            cv2.circle(circled, (int(cx), int(cy)), radius, (255, 0, 0), 2)
                        st.subheader(f"Detected MA regions — {len(contours)} cluster(s) found")
                        st.image(circled, caption="Red circles = MA clusters", use_column_width=True)

                except Exception as e:
                    st.error(f"Model inference failed: {e}")

    st.markdown("---")
    st.caption(
        "Tip: Place modules under ./processing and models under ./models."
    )


if __name__ == "__main__":
    main()