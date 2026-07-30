"""
Build a distributable Tunnel Visualization executable into ../release/.

Usage (from source/):
    python build_exe.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RELEASE = PROJECT / "release"
WORK = PROJECT / "build" / "pyinstaller"


def _ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("Installing PyInstaller…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])


def build_executable() -> None:
    print("Building Tunnel Visualization executable…")
    print("-" * 50)
    _ensure_pyinstaller()

    RELEASE.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    # Absolute path — --add-data is resolved relative to --specpath otherwise.
    readme_data = f"{ROOT / 'README.md'}{os.pathsep}."

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--console",
        "--name=TunnelVisualization",
        f"--distpath={RELEASE}",
        f"--workpath={WORK}",
        f"--specpath={PROJECT / 'build'}",
        f"--add-data={readme_data}",
        "--collect-all=dash",
        "--collect-all=plotly",
        "--collect-all=rs3",
        "--collect-all=grpc",
        "--collect-submodules=tunnel_viz",
        "--hidden-import=tunnel_viz",
        "--hidden-import=tunnel_viz.web_app",
        "--hidden-import=tunnel_viz.plotting",
        "--hidden-import=tunnel_viz.rs3_extract",
        "--hidden-import=tunnel_viz.rs3_bootstrap",
        "--hidden-import=tunnel_viz.session",
        "--hidden-import=tunnel_viz.result_types",
        "--hidden-import=plotly.graph_objs",
        "--hidden-import=plotly.subplots",
        "run_tunnel_viewer.py",
    ]

    print(f"Running:\n  {' '.join(cmd)}\n")
    print(f"Output folder: {RELEASE}")
    print("-" * 50)

    subprocess.check_call(cmd, cwd=ROOT)

    shutil.copy2(ROOT / "README.md", RELEASE / "README.md")
    (RELEASE / "HOW_TO_RUN.txt").write_text(
        "\n".join(
            [
                "Tunnel Liner Visualization",
                "==========================",
                "",
                "1. Double-click TunnelVisualization.exe",
                "2. A console window opens and your browser goes to http://127.0.0.1:8052",
                "3. Browse to a computed .rs3v3 (sibling .rs3compute required)",
                "4. Choose field / coordinate system / stage, then Extract from RS3",
                "",
                "Requirements:",
                "- RS3 installed (scripting used to extract liner results)",
                "- Windows 64-bit",
                "",
                "Optional command-line:",
                '  TunnelVisualization.exe "C:\\path\\to\\model.rs3v3" --auto',
                "",
                "Close the console window (or Ctrl+C) to stop the viewer.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    exe = RELEASE / "TunnelVisualization.exe"
    if not exe.is_file():
        raise SystemExit(f"Build finished but executable not found: {exe}")

    size_mb = exe.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 50)
    print("[SUCCESS] Build complete")
    print("=" * 50)
    print(f"  {exe}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Also copied: README.md, HOW_TO_RUN.txt")
    print("=" * 50)


if __name__ == "__main__":
    build_executable()
