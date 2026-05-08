"""download_fonts.py
====================
Downloads Noto Sans fonts for multilingual PDF report generation.
Run once: python download_fonts.py

Fonts saved to: fonts/
"""

import os
import urllib.request

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
os.makedirs(FONTS_DIR, exist_ok=True)

_BASE = "https://github.com/notofonts/noto-fonts/raw/main/hinted/ttf"

FONTS = {
    "NotoSans-Regular.ttf":
        f"{_BASE}/NotoSans/NotoSans-Regular.ttf",
    "NotoSansDevanagari-Regular.ttf":
        f"{_BASE}/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf",
    "NotoSansOdia-Regular.ttf":
        f"{_BASE}/NotoSansOriya/NotoSansOriya-Regular.ttf",
    "NotoSansBengali-Regular.ttf":
        f"{_BASE}/NotoSansBengali/NotoSansBengali-Regular.ttf",
    "NotoSansTelugu-Regular.ttf":
        f"{_BASE}/NotoSansTelugu/NotoSansTelugu-Regular.ttf",
    "NotoSansTamil-Regular.ttf":
        f"{_BASE}/NotoSansTamil/NotoSansTamil-Regular.ttf",
    "NotoSansGujarati-Regular.ttf":
        f"{_BASE}/NotoSansGujarati/NotoSansGujarati-Regular.ttf",
    "NotoSansOlChiki-Regular.ttf":
        f"{_BASE}/NotoSansOlChiki/NotoSansOlChiki-Regular.ttf",
}


def download_all():
    print(f"Downloading fonts to: {FONTS_DIR}\n")
    for filename, url in FONTS.items():
        dest = os.path.join(FONTS_DIR, filename)
        if os.path.exists(dest):
            print(f"  already exists: {filename}")
            continue
        print(f"  Downloading {filename}...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            size_kb = os.path.getsize(dest) // 1024
            print(f"done ({size_kb} KB)")
        except Exception as e:
            print(f"FAILED: {e}")
    print("\nDone. Run main.py to start the app.")


if __name__ == "__main__":
    download_all()
