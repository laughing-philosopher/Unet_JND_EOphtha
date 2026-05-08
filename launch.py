"""launch.py — Aakhi USB launcher
==================================
Double-click or run with: python launch.py
Starts the Flask server and opens the browser automatically.
Works from USB drives and any directory — paths are resolved relative to this file.
"""

import os
import socket
import sys
import threading
import time
import webbrowser


def _find_free_port(start: int = 5050) -> int:
    """Find a free TCP port starting from `start`."""
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
    return start


def main():
    # ── Ensure working directory is the script's directory ─────────────── #
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    sys.path.insert(0, script_dir)

    port = _find_free_port(5050)
    url  = f"http://localhost:{port}"

    print("=" * 55)
    print("  AAKHI — Retinal Image Analysis")
    print("  IIT Bhubaneswar · Eye AI Lab")
    print("=" * 55)
    print(f"\n  Starting on {url}")
    print("  Press Ctrl+C to stop.\n")

    # Open browser after a short delay
    def _open_browser():
        time.sleep(2.5)
        webbrowser.open(url)

    threading.Thread(target=_open_browser, daemon=True).start()

    # Import and run the Flask app
    os.environ["PORT"] = str(port)
    from main import app
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
