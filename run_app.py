"""Convenience launcher for the Streamlit UI.

Usage:
    python run_app.py
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    # headless=false helps desktop environments open a browser tab automatically.
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.address",
        "0.0.0.0",
        "--server.port",
        "8501",
        "--server.headless",
        "false",
        "--browser.serverAddress",
        "localhost",
        "--browser.gatherUsageStats",
        "false",
    ]
    print("Starting Streamlit app...")
    print("Open this in your browser: http://localhost:8501")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
