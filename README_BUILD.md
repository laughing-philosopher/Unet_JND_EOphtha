# Building the Aakhi Portable EXE

## How to build

```
build_exe.bat
```

This runs PyInstaller with `aakhi.spec` and produces `dist\Aakhi\`.

## What you get

```
dist\
  Aakhi\
    Aakhi.exe          ← double-click to launch
    models\            ← all ML model files
    static\            ← CSS / JS / i18n
    templates\         ← HTML
    fonts\             ← Noto TTFs for PDF
    processing\        ← Python modules (bundled)
    *.dll              ← TensorFlow / PyTorch / CUDA runtime
```

Zip the entire `dist\Aakhi\` folder and copy to USB.  
On any Windows 10/11 PC, extract the zip, double-click `Aakhi.exe`.

## Expected size

| Component | Approx size |
|---|---|
| TensorFlow + Keras | ~2 GB |
| PyTorch + torchvision | ~1.5 GB |
| OpenCV, PIL, other deps | ~400 MB |
| ML model files | ~800 MB |
| App code + assets | ~50 MB |
| **Total** | **~5–6 GB** |

A 16 GB USB drive is enough.

## Expected startup time

First launch: 45–90 seconds (TF + PyTorch initialise).  
Subsequent launches on same machine: 20–40 seconds (OS file cache helps).

## Known limitations

- GPU acceleration is **not** available in the bundled exe (Windows TF CPU only).  
  Inference is CPU-bound — analysis of one image takes ~2–5 minutes.
- The console window stays open — this is intentional so you can see status messages.
- If Windows Defender flags the exe, add `dist\Aakhi\` to the exclusion list  
  (PyInstaller-built exes are occasionally flagged as false positives).

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Failed to load model" | Make sure `dist\Aakhi\models\` contains all `.pth` / `.h5` / `.keras` files |
| Browser doesn't open | Navigate manually to `http://localhost:5050` |
| DLL error on startup | Install [Visual C++ Redistributable 2022](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| Port 5050 already in use | `launch.py` auto-selects a free port — check the console output |
