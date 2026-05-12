# Aakhi — Unified Retinal Image Analysis Platform

**Aakhi** is an advanced, comprehensive ophthalmic image analysis application developed for automated screening, diagnostic grading, and multi-lesion segmentation of retinal fundus images. Designed for deployment in clinical environments, mobile screening camps, and academic research labs, Aakhi integrates state-of-the-art deep learning architectures into unified, user-friendly graphical interfaces.

---

## 🌟 Key Features

- **Multi-Pipeline Diagnostics**: Supports Microaneurysm (MA) detection, Optic Disc/Cup (OD-OC) segmentation, Retinal Nerve Fiber Layer Defect (RFNLD) detection, Diabetic Retinopathy (DR) severity grading, and multi-class Lesion segmentation.
- **Granular Detection Modes**: Features dual-mode analysis (Maximum vs. Minimum modes) for Microaneurysm detection, providing clear visual and quantitative comparisons between baseline model sensitivity and heuristic-filtered refinement.
- **Multilingual Support**: Fully localized interface and PDF export capabilities supporting English, Hindi, Odia, and Bengali for expanded regional accessibility.
- **Automated Clinical Reporting**: Generates highly detailed, multi-page clinical PDF reports incorporating automated patient demographics, model predictions, visual overlays, multi-lesion color legends, and official institutional branding.
- **Portable Standalone Executable**: Fully bundled deployment support enabling zero-dependency execution directly from USB drives on any Windows 10/11 system.

---

## 🚀 How to Run the Portable Standalone Executable (`.exe`)

The portable distribution of Aakhi packages all Python interpreters, core machine learning runtimes (TensorFlow/PyTorch), web servers, UI components, and pre-trained model weights into a single standalone directory requiring **zero installation or administrative privileges**.

### Instructions:
1. **Acquire the Package**: Retrieve the pre-built application folder (`dist\Aakhi\`), typically distributed as a compressed zip archive.
2. **Extract**: Unzip the folder onto your local disk or external USB drive on any standard Windows 10/11 PC.
3. **Launch**: Navigate inside the extracted `Aakhi` folder and double-click **`Aakhi.exe`**.
4. **Startup Behavior**: 
   - A terminal console window will open displaying backend initialization logs. 
   - *Note*: Initial load times take approximately **45–90 seconds** as internal PyTorch and TensorFlow execution engines load into memory. Subsequent launches on the same system are significantly faster due to OS file caching.
   - Once initialized, the application will automatically select an available local network port and launch the primary interface directly inside your default web browser.
5. **Manual Navigation**: If your browser blocks automatic pop-ups, locate the local access address displayed in the background console window (e.g., `http://localhost:5050` or `http://127.0.0.1:5050`) and manually enter it into your web browser.

> **⚠️ Known Limitations**: The bundled portable `.exe` runs inference entirely on the CPU to ensure maximum plug-and-play compatibility across diverse standard clinical hardware. Processing complex fundus images through multiple concurrent deep learning models may take 1 to 4 minutes per scan. Please ensure the background terminal console remains open while the application is in use.

---

## 📂 Project File Structure & Usage Guide

### 🖥️ Main Interfaces & Application Entry Points
| File | Role & Usage Description |
| :--- | :--- |
| **`app.py`** | The primary **Streamlit Web Application** interface. Orchestrates file uploading, interactive patient demographic entry, real-time side-by-side mode comparisons, dynamic threshold tuning, localized translations, and PDF export compilation. |
| **`main.py`** | The **Flask Web Application Backend**. Provides robust API endpoints, multi-threaded background task orchestration, real-time progress streaming via Server-Sent Events (SSE), and alternative web template rendering. |
| **`streamlit.py`** | The **Master Lab UI**. Tailored specifically for the Image and Video Processing Lab (IVP Lab), offering dedicated, independent model inspection tabs, manual point-click coordinate selectors, and research-focused configurations. |
| **`launch.py`** | **Portable Launcher Script**. Automatically discovers a free TCP port, configures environment paths, initializes the Flask execution engine (`main.py`), and invokes the system's default web browser. |
| **`app_launcher.py`** | Programmatic headless launcher for embedding or invoking the standard Streamlit interface without external terminal commands. |

### 🧠 Deep Learning Processing Modules (`processing/`)
| File | Role & Usage Description |
| :--- | :--- |
| **`processing_ma.py`** | Core **Microaneurysm Detection Pipeline** utilizing UNet models. Exposes both *Maximum Mode* (high-sensitivity candidate mask) and *Minimum Mode* (heuristic post-processing filtering candidates by area, circularity, and confidence to eliminate false positives). |
| **`processing_odoc.py`** | **Optic Disc & Optic Cup Segmentation** pipeline. Derives boundary outlines, calculates the vertical Cup-to-Disc Ratio (vCDR), and includes an integrated Random Forest classifier for preliminary Glaucoma risk evaluation. |
| **`processing_odoc_basic.py`** | Core/lightweight variant of the OD-OC segmentation logic. |
| **`processing_rfnld.py`** | **Retinal Nerve Fiber Layer Defect Detection** pipeline. Accepts either manual disc center/rim point clicks or automatically derives required ring coordinates directly from ODOC segmentation metrics. |
| **`processing_dr_grading.py`** | Loads an EfficientNet-B6 architecture to grade input images into standard **Diabetic Retinopathy Severity** categories (No DR, Mild, Moderate, Severe, Proliferative). |
| **`processing_glaucoma_grading.py`** | Deep learning inference module evaluating image-level **Glaucoma Severity** using an optimized EfficientNet backbone. |
| **`processing_lesion.py`** | **Multi-Lesion Detection Pipeline** utilizing a UNet + FIAM model trained on IDRiD. Produces color-coded multi-class pixel maps mapping Hard Exudates (Red), Hemorrhages (Green), Microaneurysms (Blue), and Soft Exudates (Yellow). |
| **`overlay_odoc.py`** | Rendering utility script building high-fidelity visual segmentation overlays and calculating exact disc rim and cup structural metrics. |

### 📄 Clinical Reporting & Core Services
| File | Role & Usage Description |
| :--- | :--- |
| **`report_v2.py`** | The advanced **ReportLab PDF Generator**. Constructs professional, multi-page clinical reports featuring customized branding, native localized scripts, structured summary data tables, embedded mode comparisons, and discrete visual section layouts. |
| **`report.py`** | Standard/legacy PDF report compilation logic used by the primary Streamlit interface. |
| **`translations.py`** | Centralized **i18n Localization Dictionary**. Manages UI string translations and font mapping logic across English, Hindi, Odia, and Bengali. |
| **`auth.py`** | Session management and authentication module providing secure access controls, local credential validation, and role assignment (Doctor/Admin/Guest). |
| **`functions.py` / `helper.py`** | Shared utilities for file paths, base64 image encoding/decoding, array scaling, and graphical channel blending. |
| **`download_fonts.py`** | Automated bootstrapping script for fetching and locally caching official Google Noto TrueType fonts required for multi-script Unicode PDF generation. |

### 📦 Build Automation & Configuration Files
| File | Role & Usage Description |
| :--- | :--- |
| **`build_exe.bat`** | Windows Batch build script. Automatically locates the active virtual environment PyInstaller binary, cleans build artifacts, and executes the compilation specification to build the portable standalone release folder. |
| **`aakhi.spec`** | Advanced **PyInstaller Bundling Specification**. Explicitly defines bundled data trees, nested static CSS/JS folders, Jinja2 HTML templates, translation files, model paths, and binary dynamic runtime dependencies (`.dll` files). |
| **`aakhi_hook.py`** | Custom **PyInstaller Runtime Hook**. Injected at initialization to dynamically redirect `AAKHI_BASE_PATH` references to the temporary execution directory (`sys._MEIPASS`), guaranteeing flawless file and model resolution inside the frozen `.exe`. |
| **`pyinstaller.spec`** | Alternative or baseline PyInstaller build configuration specification. |
| **`install.bat`** | Deployment environment script designed for setting up local embeddable Python runtimes. Enables site-packages, fetches `pip`, and installs CPU-optimized native builds of ML frameworks. |
| **`run_aakhi.bat`** | Direct start script for local source-based execution using a portable embeddable Python folder. |

### 🗂️ Data & Static Resource Folders
- **`models/`**: Houses all core pre-trained deep learning weight files (`.h5`, `.bin`, `.pth`).
- **`static/` / `templates/`**: Frontend user interface stylesheets, client-side scripts, localized JSON dictionaries, and Flask Jinja2 web layout files.
- **`fonts/`**: Cached TrueType font binaries ensuring clean cross-platform vector text rendering in exported PDF documents.
- **`users.json`**: Flat-file database containing local registered user accounts and permission settings.

---

## 🛠️ Typical Development Setup & Local Execution

For local source-based modification or testing outside the pre-compiled `.exe` bundle:

1. **Clone & Setup Virtual Environment**:
   ```cmd
   python -m venv venv
   call venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Download Required Assets**: Ensure all model weights are placed inside the `models/` directory and run `python download_fonts.py` to prepare local typography assets.
3. **Launch Streamlit Frontend**:
   ```cmd
   streamlit run app.py
   ```
4. **Launch Flask Backend / API API**:
   ```cmd
   python launch.py
   ```