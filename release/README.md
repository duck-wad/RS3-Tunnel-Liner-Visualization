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

- Python 3.10+
- RS3 with scripting (`RS3Scripting`, matched to your RS3 version)
- Packages in `requirements.txt`
