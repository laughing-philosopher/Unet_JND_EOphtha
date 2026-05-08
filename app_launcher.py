import os
import sys
from streamlit.web.cli import main as st_main


def main() -> None:
	# Resolve path to the Streamlit app file next to this launcher
	app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit.py")

	# Prefer a free random port and bind explicitly to localhost
	# This helps when 8501 is already in use or blocked
	env = os.environ
	env.setdefault("STREAMLIT_SERVER_PORT", "0")  # 0 => auto-select free port
	env.setdefault("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")
	env.setdefault("STREAMLIT_BROWSER_GATHERUSAGESTATS", "false")

	# Build argv for `streamlit run`
	sys.argv = [
		"streamlit",
		"run",
		app_path,
		"--server.headless=true",
		"--server.address=127.0.0.1",
		"--server.port=0",
		"--browser.gatherUsageStats=false",
	]

	# Invoke Streamlit programmatically
	st_main()


if __name__ == "__main__":
	main()


