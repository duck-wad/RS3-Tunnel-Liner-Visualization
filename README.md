# Tunnel Liner Visualization

Interactive 2D/3D tunnel visualization for liner results (displacement, moment, axial force), pulled directly from a computed RS3 model via scripting.

## Web viewer

```bash
cd source
pip install -r requirements.txt
python run_tunnel_viewer.py
```

1. Click **Browse** and select a computed `.rs3v3` (sibling `.rs3compute` required).
2. Choose a liner result (e.g. Z Displacement, Moment Y, Axial Force X) and Global/Local axes.
3. Click **Extract from RS3** — liner stages are detected, results are pulled, and the Plotly view updates.
4. Use the **Stage** dropdown to switch among stages that have liners.

```bash
python run_tunnel_viewer.py "path\to\model.rs3v3" --auto --field moment_y --coord Local
```

- `--port` — RS3 scripting port (default `60064`)
- `--http-port` — local Dash server (default `8052`)
- `--no-start` — do not launch RS3 if no scripting server is running
- `--field` — `displacement_z`, `axial_force_x`, `moment_y`, …

## Requirements

- Python 3.10+ (for development / building the exe)
- RS3 with scripting (`RS3Scripting`, matched to your RS3 version)
- Packages in `requirements.txt`

## Windows executable

### Build

Requires Python with the project dependencies and PyInstaller installed:

```bash
cd source
pip install -r requirements.txt
pip install pyinstaller
python build_exe.py
```

This writes:

- `release/TunnelVisualization.exe`
- `release/HOW_TO_RUN.txt`
- `release/README.md`

The `.exe` is gitignored (too large for GitHub). Keep it local or attach it to a [GitHub Release](https://docs.github.com/en/repositories/releasing-projects-on-github).

### Run (end users)

1. Double-click `release/TunnelVisualization.exe`  
   — or from a terminal: `release\TunnelVisualization.exe`
2. A console window opens and the browser goes to `http://127.0.0.1:8052`
3. Browse to a computed `.rs3v3`, choose field/coords/stage, then **Extract from RS3**

RS3 must be installed on the machine; Python is not required to run the exe.

Optional CLI (same flags as the Python launcher):

```bash
TunnelVisualization.exe "C:\path\to\model.rs3v3" --auto
```

Close the console window (or Ctrl+C) to stop the viewer.
