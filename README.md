# Unet_JND_EOphtha

## Overview

Unet_JND_EOphtha is a Streamlit-based ophthalmic image analysis application for retinal fundus images.
It combines multiple deep-learning workflows in one interface so users can upload a fundus image and run different diagnostic pipelines.

The app currently supports:

- Microaneurysm (MA) detection
- Optic Disc / Optic Cup (OD-OC) segmentation
- RNFL defect (RFNLD) detection
- Diabetic Retinopathy (DR) severity grading

## Model Input/Output Map

| Model | Input | User Action | Output in UI |
|---|---|---|---|
| MA (Microaneurysm) | RGB fundus image | Set threshold and batch size, then run | Probability/mask output and green overlay on the image |
| ODOC (Optic Disc / Cup) | RGB fundus image | Set threshold and batch size, then run | Segmentation-style mask/probability output and overlay view |
| RFNLD | RGB fundus image | Click disc center (C), then click rim point (R), then run | Fundus image with detected RNFL defect line(s) drawn |
| DR Grading (DRG) | RGB fundus image | Run DR grading | Predicted DR severity class (No DR, Mild, Moderate, Severe, Proliferative DR) |

## Typical Workflow

1. Launch the Streamlit app.
2. Select one of the available models from the sidebar.
3. Upload a retinal fundus image.
4. Provide model-specific inputs (for example threshold/batch, or RFNLD click points).
5. Run inference and inspect model outputs.

## Project Structure (Key Files)

- app.py: Main Streamlit UI and model orchestration
- processing/processing_ma.py: MA pipeline
- processing/processing_odoc.py: OD-OC pipeline
- processing/processing_rnfld.py: RFNLD pipeline
- processing/processing_dr_grading.py: DR grading model loading and prediction
- models/: Saved model weights