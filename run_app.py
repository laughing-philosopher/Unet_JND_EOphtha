import os
import sys
import shutil
import tempfile

# Ensure the app root (next to this file) is on sys.path so relative imports work
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

# Prefer invoking Streamlit via its CLI entrypoint (works well with PyInstaller)
from streamlit.web.cli import main as stcli

# Copy the bundled app.py to a writable temp location to avoid permission issues
src_script_path = os.path.join(APP_ROOT, "app.py")
if not os.path.exists(src_script_path):
    raise FileNotFoundError(f"Cannot find Streamlit entry at {src_script_path}")

# Default: try to run from a writable copy; on failure, fall back to src path
try:
    temp_dir = tempfile.mkdtemp(prefix="unet_jnd_eophtha_")
    SCRIPT_PATH = os.path.join(temp_dir, "app.py")
    shutil.copyfile(src_script_path, SCRIPT_PATH)
except PermissionError:
    # Fall back to the original path inside the bundle (read-only is fine for Streamlit)
    SCRIPT_PATH = src_script_path

# Configure argv similar to: streamlit run app.py --server.headless=true
sys.argv = [
    "streamlit",
    "run",
    SCRIPT_PATH,
    "--server.headless=true",
    "--server.port=8501",
    "--server.address=127.0.0.1",
    "--global.developmentMode=false",
]

sys.exit(stcli())
