# aakhi.spec — PyInstaller build spec for Aakhi Retinal Analysis
# Build:  pyinstaller aakhi.spec
# Output: dist\Aakhi\Aakhi.exe  (+ all support files in dist\Aakhi\)
# Ship:   zip dist\Aakhi\ and copy to USB

import sys
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

block_cipher = None

# ── Collect heavy packages ──────────────────────────────────────────────── #
# Each collect_all() returns (datas, binaries, hiddenimports)

tf_d,  tf_b,  tf_h  = collect_all('tensorflow')
k_d,   k_b,   k_h   = collect_all('keras')
pt_d,  pt_b,  pt_h  = collect_all('torch')
tv_d,  tv_b,  tv_h  = collect_all('torchvision')
smp_d, smp_b, smp_h = collect_all('segmentation_models_pytorch')
cv_d,  cv_b,  cv_h  = collect_all('cv2')
pil_d, pil_b, pil_h = collect_all('PIL')
rl_d,  rl_b,  rl_h  = collect_all('reportlab')
sk_d,  sk_b,  sk_h  = collect_all('sklearn')

all_datas = (
    tf_d + k_d + pt_d + tv_d + smp_d + cv_d + pil_d + rl_d + sk_d +
    # App assets — (src_path, dest_in_bundle)
    [
        ('models',           'models'),
        ('static',           'static'),
        ('templates',        'templates'),
        ('fonts',            'fonts'),
        ('processing',       'processing'),
        ('main.py',          '.'),
        ('auth.py',          '.'),
        ('report_v2.py',     '.'),
        ('aakhi_logo.png',   '.'),
        ('iitbbs logo.png',  '.'),
    ]
)

all_binaries = tf_b + k_b + pt_b + tv_b + smp_b + cv_b + pil_b + rl_b + sk_b

all_hidden = (
    tf_h + k_h + pt_h + tv_h + smp_h + cv_h + pil_h + rl_h + sk_h +
    [
        # Flask ecosystem
        'flask', 'flask.templating', 'werkzeug', 'jinja2',
        'click', 'itsdangerous',
        # Data
        'numpy', 'scipy', 'pandas',
        # Image
        'PIL', 'PIL.Image', 'PIL.ImageDraw', 'PIL.ImageFont',
        'cv2',
        # ML
        'joblib', 'sklearn', 'sklearn.ensemble', 'sklearn.preprocessing',
        # PDF
        'reportlab', 'reportlab.pdfbase', 'reportlab.pdfbase.ttfonts',
        'reportlab.platypus', 'reportlab.lib',
        # App modules
        'processing', 'processing.processing_dr_grading',
        'processing.processing_glaucoma_grading',
        'processing.processing_odoc',
        'processing.processing_odoc_basic',
        'processing.processing_lesion',
        'processing.processing_ma',
        'processing.processing_rfnld',
        'processing.overlay_odoc',
        'auth', 'report_v2', 'translations', 'helper',
        # efficientnet (TF DR model)
        'efficientnet', 'efficientnet.tfkeras',
        # misc
        'uuid', 'queue', 'threading', 'tempfile', 'traceback',
    ]
)

a = Analysis(
    ['launch.py'],
    pathex=['.'],
    binaries=all_binaries,
    datas=all_datas,
    hiddenimports=all_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['aakhi_hook.py'],
    excludes=[
        # Trim unused heavy packages to shrink output
        'matplotlib', 'notebook', 'IPython', 'ipykernel',
        'pytest', 'sphinx', 'docutils',
        'tkinter', 'wx', 'PyQt5', 'PyQt6',
        # Audio — not used, DLL version mismatch on some installs
        'torchaudio',
        # Optional backends we don't need
        'openvino', 'tensorboard', 'torch.utils.tensorboard',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Aakhi',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX can corrupt TF/PyTorch DLLs — leave off
    console=True,       # Keep console so startup messages are visible
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Aakhi',
)
