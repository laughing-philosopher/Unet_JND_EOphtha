import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import os

# --- Optional: coordinate picker for RFNLD ---
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_COORD_PICKER = True
except Exception:
    HAS_COORD_PICKER = False

# Import processing modules (update paths/names if yours differ)
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


# Utility: overlay mask on an image (green channel)
def overlay_mask_on_rgb(rgb_img, mask, alpha=0.6):
    # rgb_img: HxWx3 uint8
    # mask: HxW boolean/0-1
    out = rgb_img.copy()
    mask_bool = (mask > 0)
    out[mask_bool, 1] = 255
    return out


# Model metadata (RHS quick info / description)
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
        "recommended_threshold": None,  # not used
        "recommended_batch": None,      # not used
        "notes": (
            "This model does not use threshold/batch. It needs two clicks to define the ROI: "
            "first click = disc center (C), second click = rim point (R)."
        ),
    },
}


def set_selected(model_key: str):
    st.session_state.selected_model = model_key
    st.session_state.sidebar_run_click = True


def main():
    st.set_page_config(layout="wide", page_title="Unet_JND_EOphtha - Master")

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "MA"
    if "sidebar_run_click" not in st.session_state:
        st.session_state.sidebar_run_click = False

    # Sidebar (left) — model selector + quick info
    with st.sidebar:
        st.markdown("## Models (click to select & run)")
        if st.button("Microaneurysm (MA)"):
            set_selected("MA")
        if st.button("OD - OC Segmentation (ODOC)"):
            set_selected("ODOC")
        if st.button("RFNLD Detector (RFNLD)"):
            set_selected("RFNLD")

        st.markdown("---")
        st.markdown("### Quick info")
        info = MODEL_INFO[st.session_state.selected_model]
        st.write("Selected model:", info["title"]) 
        st.markdown(info["notes"])

    # Header with logo on the right
    col_left, col_right = st.columns([3, 1])
    with col_left:
        st.title("Image and Video Processing Lab")
        st.subheader("Unet_JND_EOphtha — Master UI")
        st.markdown(f"**Active model:** {MODEL_INFO[st.session_state.selected_model]['title']}")
    with col_right:
        logo_path = "iitbbs logo.png"
        if os.path.exists(logo_path):
            st.image(Image.open(logo_path), width=200)

    st.markdown("---")

    # Split main content: center workspace + RHS description
    main_col, info_col = st.columns([3, 1])

    # RHS — description always visible
    with info_col:
        info = MODEL_INFO[st.session_state.selected_model]
        st.header(info["title"])
        st.write("**Description:**")
        st.write(info["description"]) 
        if st.session_state.selected_model != "RFNLD":
            st.write("**Recommended threshold:**", info["recommended_threshold"]) 
            st.write("**Recommended batch size:**", info["recommended_batch"]) 
        else:
            st.write("**Threshold / batch:** Not used for this model.")
        st.write("**Notes:**")
        st.write(info["notes"]) 

    # Center — per-model UI
    with main_col:
        model_key = st.session_state.selected_model
        info = MODEL_INFO[model_key]

        uploaded = st.file_uploader("Upload fundus image", type=["jpg", "jpeg", "png"], key=f"uploader_{model_key}")

        if uploaded is None:
            st.info("Please upload an image to run the selected model.")
            return

        # Load as RGB numpy
        image_pil = Image.open(uploaded).convert("RGB")
        image_cv2 = np.array(image_pil)

        st.subheader("Input image")
        st.image(image_cv2, use_column_width=True)

        # --- RFNLD: ROI clicks flow ---
        if model_key == "RFNLD":
            if not HAS_COORD_PICKER:
                st.error(
                    "streamlit-image-coordinates is required. Install with: \n"
                    "pip install streamlit-image-coordinates"
                )
                return

            st.markdown("**Step 1:** Click the *disc center (C)* on the image below.")
            click_c = streamlit_image_coordinates(image_pil, key="click_center")

            if click_c:
                cx, cy = click_c["x"], click_c["y"]
                # draw the first point for step 2 visualization
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
                            st.error("RFNLD processing module not available or missing `processing` function.")
                            return
                        processing_fn = getattr(info["module"], "processing")

                        # Build the coordinates dicts expected by your function
                        coord_center = {"x": int(cx), "y": int(cy)}
                        coord_rim = {"x": int(rx), "y": int(ry)}

                        with st.spinner("Running RFNLD Detector ..."):
                            try:
                                # Your RFNLD `processing` returns a BGR/ RGB image with lines drawn
                                out_img = processing_fn(image_cv2, coord_rim, coord_center)

                                if out_img is None:
                                    st.error("Model returned None — check the processing function.")
                                    return

                                # Ensure RGB for display
                                out_np = np.array(out_img)
                                if out_np.ndim == 3 and out_np.shape[2] == 3:
                                    # If it's BGR (OpenCV), convert to RGB for Streamlit. Heuristic: assume BGR.
                                    out_np = cv2.cvtColor(out_np, cv2.COLOR_BGR2RGB)

                                st.subheader("RFNLD output")
                                st.image(out_np, use_column_width=True)
                            except Exception as e:
                                st.error(f"Model inference failed: {e}")
            return  # RFNLD path finishes here

        # --- MA / OD-OC paths (threshold + batch) ---
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
                st.error("Processing module not available or missing `processing` function.")
                return

            processing_fn = getattr(info["module"], "processing")

            with st.spinner(f"Running {info['title']} ..."):
                try:
                    result = processing_fn(image_cv2, float(thr), int(batch))

                    # If model returns a scalar score, just display the score
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
                except Exception as e:
                    st.error(f"Model inference failed: {e}")

    st.markdown("---")
    st.caption(
        "Tip: Place modules under ./processing (processing_ma.py, processing_odoc.py, processing_rnfld.py) "
        "and models under ./models."
    )


if __name__ == "__main__":
    main()
