# aakhi_hook.py — PyInstaller runtime hook
# Sets AAKHI_BASE_PATH so all modules can find data files
# regardless of whether running as script or bundled exe.
import os
import sys

if hasattr(sys, '_MEIPASS'):
    os.environ['AAKHI_BASE_PATH'] = sys._MEIPASS
else:
    os.environ['AAKHI_BASE_PATH'] = os.path.dirname(os.path.abspath(sys.argv[0]))
