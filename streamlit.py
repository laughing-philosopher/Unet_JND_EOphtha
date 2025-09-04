import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os

# Import processing modules
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
    out = rgb_img.copy()
    mask_bool = (mask > 0)
    out[mask_bool, 1] = 255
    return out


# Model metadata
MODEL_INFO = {
    "MA": {
        "title": "Microaneurysm Detector (MA)",
        "module": proc_ma,
        "description": "Detects microaneurysms in retinal images. Input: RGB fundus image. Output: probability/PSM map and binary mask of candidate microaneurysms.",
        "recommended_threshold": 0.9,
        "recommended_batch": 20,
        "notes": "Uses UNet + patch-based inference pipeline."
    },
    "ODOC": {
        "title": "Optic Disc / Optic Cup (OD-OC) Segmentation",
        "module": proc_od,
        "description": "Segments optic disc and optic cup from retinal fundus images. Useful for glaucoma analysis.",
        "recommended_threshold": 0.5,
        "recommended_batch": 8,
        "notes": "Requires processing_odoc.py exposing `processing(image, threshold, batch_size)`."
    },
    "RFNLD": {
        "title": "RNFLD Detector (Retinal Nerve Fiber Layer Defect)",
        "module": proc_rfnld,
        "description": "Detects retinal nerve fiber layer defects. Input: RGB fundus image. Output: lesion probability map / binary mask.",
        "recommended_threshold": 0.6,
        "recommended_batch": 16,
        "notes": "Requires processing_rnfld.py exposing `processing(image, threshold, batch_size)`."
    }
}


def set_selected(model_key):
    st.session_state.selected_model = model_key
    st.session_state.sidebar_run_click = True


def main():
    st.set_page_config(layout="wide", page_title="Unet_JND_EOphtha - Master")

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "MA"
    if "sidebar_run_click" not in st.session_state:
        st.session_state.sidebar_run_click = False

    # Sidebar model selection
    with st.sidebar:
        st.markdown("## Models (click to select & run)")
        if st.button("Microaneurysm (MA)"):
            set_selected("MA")
        if st.button("OD - OC Segmentation (ODOC)"):
            set_selected("ODOC")
        if st.button("RNFLD Detector (RFNLD)"):
            set_selected("RFNLD")

        st.markdown("---")
        st.markdown("### Quick info")
        st.write("**Selected model:**", MODEL_INFO[st.session_state.selected_model]["title"])
        st.info(MODEL_INFO[st.session_state.selected_model]["notes"])

    # Header with logo
    col_left, col_right = st.columns([3, 1])
    with col_left:
        st.title("Image and Video Processing Lab")
        st.subheader("Unet_JND_EOphtha — Master UI")
        st.markdown(f"**Active model:** {MODEL_INFO[st.session_state.selected_model]['title']}")
    with col_right:
        logo_path = "iitbbs logo.png"
        if os.path.exists(logo_path):
            logo = Image.open(logo_path)
            st.image(logo, width=200)

    st.markdown("---")

    # Split main content
    main_col, info_col = st.columns([3, 1])

    # RHS: always show description + quick info
    with info_col:
        info = MODEL_INFO[st.session_state.selected_model]
        st.header(info["title"])
        st.write("**Description:**")
        st.write(info["description"])
        st.write("**Recommended threshold:**", info["recommended_threshold"])
        st.write("**Recommended batch size:**", info["recommended_batch"])
        st.write("**Notes:**")
        st.write(info["notes"])

    # Center: controls + results
    with main_col:
        info = MODEL_INFO[st.session_state.selected_model]
        thr = st.number_input(
            "Threshold (probability cutoff)",
            min_value=0.0, max_value=1.0,
            value=float(info["recommended_threshold"]),
            step=0.01,
            key="threshold_input"
        )
        batch = st.number_input(
            "Batch size",
            min_value=1, max_value=256,
            value=int(info["recommended_batch"]),
            step=1,
            key="batch_input"
        )

        uploaded = st.file_uploader("Upload fundus image", type=["jpg", "jpeg", "png"])
        run_now = st.button("Run model on uploaded image")

        trigger_run = False
        if uploaded is not None:
            if st.session_state.sidebar_run_click:
                trigger_run = True
                st.session_state.sidebar_run_click = False
            if run_now:
                trigger_run = True

        if uploaded is None:
            st.info("Please upload an image to run the selected model.")
        else:
            image_pil = Image.open(uploaded).convert("RGB")
            image_cv2 = np.array(image_pil)

            st.subheader("Input image")
            st.image(image_cv2, width="stretch")



            if trigger_run:
                proc_module = info["module"]
                if proc_module is None:
                    st.error("Processing module not found for this model.")
                elif not hasattr(proc_module, "processing"):
                    st.error("Processing module does not expose `processing(img, threshold, batch_size)`.")
                else:
                    processing_fn = getattr(proc_module, "processing")
                    with st.spinner(f"Running {info['title']} ..."):
                        try:
                            result_mask = processing_fn(image_cv2, float(thr), int(batch))
                            if result_mask is None:
                                st.error("Model returned None.")
                            else:
                                mask = np.array(result_mask)
                                if mask.ndim == 3 and mask.shape[-1] == 1:
                                    mask = mask[..., 0]
                                mask = mask.astype(np.float32)

                                st.subheader("Model output (mask / probability map)")
                                disp = (mask * 255.0).clip(0, 255).astype(np.uint8)
                                st.image(disp, caption="Model output (0-255)", width="stretch")

                                overlay = overlay_mask_on_rgb(image_cv2, mask > 0)
                                st.subheader("Overlay (green marks)")
                                st.image(overlay, width="stretch")

                        except Exception as e:
                            st.error(f"Model inference failed: {e}")

    st.markdown("---")
    st.caption("Tip: Place processing modules (processing_ma.py, processing_odoc.py, processing_rnfld.py) inside the processing/ folder alongside streamlit.py")


if __name__ == "__main__":
    main()