"""
Dash + Plotly web UI for tunnel liner visualization.

Browse a computed .rs3v3, extract liner results via RS3 scripting, and plot
cross-sections along the tunnel — same workflow as the FoS contour viewer.
"""

from __future__ import annotations

import os
import tempfile
import time
import traceback
import webbrowser
from pathlib import Path
from threading import Thread, Timer
from typing import Any, Optional

import pandas as pd
from dash import Dash, Input, Output, State, callback, dcc, html, no_update

from .plotting import _empty_figure, build_tunnel_figure
from .result_types import (
    CoordSystem,
    dataframe_from_bundle,
    dropdown_options,
    effective_coord_for_view,
    get_field,
)
from .rs3_extract import StageInfo

_TEMP_FILE_PATH = os.path.join(
    tempfile.gettempdir(), "tunnel_viz_selected_model.txt"
)
_DEFAULT_HTTP_PORT = 8052

_df: Optional[pd.DataFrame] = None
_bundle_df: Optional[pd.DataFrame] = None
_stage_bundles: dict[int, pd.DataFrame] = {}
_figure = None
_log_lines: list[str] = []
_no_start: bool = False
_extract_running: bool = False
_extract_started_at: float = 0.0
_extract_job_id: int = 0
_extract_finished_job: int = 0
_extract_applied_job: int = 0
_extract_error: Optional[str] = None
_pending_figure = None
_pending_status: Optional[str] = None
_pending_stage: Optional[int] = None
_pending_stage_options: list[dict[str, Any]] = []
_pending_port: Optional[int] = None
_pending_bundle: Optional[pd.DataFrame] = None
_pending_df: Optional[pd.DataFrame] = None
_liner_stages: list[StageInfo] = []
_active_stage: Optional[int] = None
_last_model_path: str = ""
_last_port: int = 60064
_last_field_key: str = "displacement_z"
_last_coord: str = "Global"


def _btn(color: str) -> dict:
    return {
        "background": color,
        "color": "white",
        "border": "none",
        "padding": "8px 12px",
        "borderRadius": "4px",
        "cursor": "pointer",
        "flex": "1",
        "fontFamily": "Segoe UI, Arial, sans-serif",
    }


def _input() -> dict:
    return {
        "width": "100%",
        "padding": "8px",
        "boxSizing": "border-box",
        "marginBottom": "8px",
        "fontFamily": "Segoe UI, Arial, sans-serif",
    }


def _labeled(label: str, child) -> html.Div:
    return html.Div(
        [
            html.Label(
                label,
                style={
                    "fontWeight": "bold",
                    "display": "block",
                    "fontFamily": "Segoe UI, Arial, sans-serif",
                },
            ),
            child,
        ],
        style={"marginBottom": "10px"},
    )


def _append_log(msg: str) -> None:
    if _extract_running and _extract_started_at:
        elapsed = time.monotonic() - _extract_started_at
        stamp = f"[{int(elapsed) // 60:d}:{int(elapsed) % 60:02d}] "
    else:
        stamp = time.strftime("[%H:%M:%S] ")
    for line in str(msg).splitlines() or [""]:
        _log_lines.append(f"{stamp}{line}")
    if len(_log_lines) > 400:
        del _log_lines[:-400]


def _log_text() -> str:
    return "\n".join(_log_lines[-80:])


def _stage_options(stages: list[StageInfo]) -> list[dict[str, Any]]:
    # Prefer stages that actually have liner properties assigned.
    use = [s for s in stages if s.has_liner] or list(stages)
    return [
        {
            "label": f"{s.number} — {s.name}",
            "value": s.number,
        }
        for s in use
    ]


def _run_file_dialog(*, title: str, filetypes, dest: str) -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        root.focus_force()
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
        if path:
            with open(dest, "w", encoding="utf-8") as f:
                f.write(path)
    except Exception as exc:  # noqa: BLE001
        _append_log(f"File dialog error: {exc}")


def _extract_from_model(
    model_path: str,
    port: int,
    field_key: str,
    coord_system: str,
    *,
    stage_number: int | None = None,
    no_start: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, int, list[StageInfo], int]:
    """Return ``(view_df, bundle_df, stage, liner_stages, port)``."""
    from .rs3_extract import extract_all_liner_results
    from .session import connect_model

    model = Path(model_path)
    field = get_field(field_key)

    def progress(msg: str) -> None:
        _append_log(msg)

    progress(f"Model: {model.name}")
    progress(
        f"View: {field.label} | Axes: {coord_system} | "
        f"Stage: {stage_number if stage_number else 'auto'} | RS3 port {port}"
    )
    progress("Connecting to RS3…")
    t0 = time.monotonic()
    session = connect_model(
        model,
        port=port,
        force_no_start=no_start,
        progress_callback=progress,
    )
    used_port = int(session.port)
    if used_port != port:
        progress(f"Using scripting port {used_port} (requested {port}).")
    progress(f"Connected in {time.monotonic() - t0:.1f}s.")
    try:
        bundle, used_stage, liner_stages = extract_all_liner_results(
            session.model,
            stage_number=stage_number,
            progress_callback=progress,
        )
        view = dataframe_from_bundle(bundle, field_key, coord_system)
        eff = effective_coord_for_view(field_key, coord_system)
        progress(
            f"Extract complete in {time.monotonic() - t0:.1f}s — "
            f"{len(bundle):,} nodes, {len(bundle.columns) - 3} cached fields "
            f"(stage {used_stage}; viewing {field.label} [{eff}])."
        )
        return view, bundle, used_stage, liner_stages, used_port
    finally:
        try:
            session.close(keep_open=True)
        except Exception as exc:  # noqa: BLE001
            progress(f"Warning during session cleanup: {exc}")


def _status_ready(
    n_points: int,
    field_key: str,
    coord_system: str,
    stage: int,
    port: int,
    *,
    from_cache: bool = False,
) -> str:
    stage_label = next(
        (s.name for s in _liner_stages if s.number == stage),
        str(stage),
    )
    field = get_field(field_key)
    eff = effective_coord_for_view(field_key, coord_system)
    note = ""
    if (
        str(coord_system) == CoordSystem.GLOBAL.value
        and eff == CoordSystem.LOCAL.value
    ):
        note = " · (forces/moments use Local axes)"
    source = "cached" if from_cache else "extracted"
    return (
        f"Ready — {n_points:,} points · {field.label} [{eff}] "
        f"· stage {stage} ({stage_label}) · {source} · port {port}{note}"
    )


def _start_extract_thread(
    model_path: str,
    port: int,
    field_key: str,
    coord_system: str,
    *,
    stage_number: int | None = None,
) -> bool:
    global _df, _bundle_df, _stage_bundles, _extract_running, _extract_started_at
    global _extract_job_id
    global _extract_error, _pending_figure, _pending_status
    global _pending_stage, _pending_stage_options, _pending_port
    global _pending_bundle, _pending_df
    global _liner_stages, _active_stage
    global _last_model_path, _last_port, _last_field_key, _last_coord

    if _extract_running:
        _append_log("Extract already in progress — ignoring duplicate click.")
        return False

    # New model → drop stage caches from the previous project.
    if model_path != _last_model_path:
        _stage_bundles = {}

    _log_lines.clear()
    _df = None
    _extract_error = None
    _pending_figure = None
    _pending_status = None
    _pending_stage = None
    _pending_stage_options = []
    _pending_port = None
    _pending_bundle = None
    _pending_df = None
    _extract_job_id += 1
    job_id = _extract_job_id
    _extract_started_at = time.monotonic()
    _extract_running = True
    _last_model_path = model_path
    _last_port = port
    _last_field_key = field_key
    _last_coord = coord_system
    _append_log("Starting bulk extraction (all fields for this stage)…")

    def worker() -> None:
        global _df, _bundle_df, _stage_bundles
        global _extract_running, _extract_finished_job, _extract_error
        global _pending_figure, _pending_status
        global _pending_stage, _pending_stage_options, _pending_port
        global _pending_bundle, _pending_df
        global _liner_stages, _active_stage, _last_port
        try:
            if not model_path or not Path(model_path).exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            view, bundle, used_stage, liner_stages, used_port = _extract_from_model(
                model_path,
                port,
                field_key,
                coord_system,
                stage_number=stage_number,
                no_start=_no_start,
            )
            if job_id != _extract_job_id:
                return
            _append_log("Building Plotly figure…")
            t_fig = time.monotonic()
            fig = build_tunnel_figure(view, progress_callback=_append_log)
            _df = view
            _bundle_df = bundle
            _stage_bundles[int(used_stage)] = bundle
            _liner_stages = liner_stages
            _active_stage = used_stage
            _last_port = used_port
            _pending_figure = fig
            _pending_df = view
            _pending_bundle = bundle
            _pending_stage = used_stage
            _pending_stage_options = _stage_options(liner_stages)
            _pending_port = used_port
            cached_stages = ", ".join(str(s) for s in sorted(_stage_bundles))
            _pending_status = _status_ready(
                len(view), field_key, coord_system, used_stage, used_port
            ) + f" · stages in memory: {cached_stages}"
            _extract_error = None
            _append_log(
                f"Figure ready in {time.monotonic() - t_fig:.1f}s. "
                f"Cached stages: {cached_stages}."
            )
        except Exception as exc:  # noqa: BLE001
            if job_id != _extract_job_id:
                return
            msg = str(exc).strip() or exc.__class__.__name__
            _append_log(msg)
            _append_log(traceback.format_exc())
            _extract_error = msg
            _df = None
            _pending_figure = None
            _pending_status = None
            _pending_stage = None
            _pending_stage_options = []
            _pending_port = None
            _pending_bundle = None
            _pending_df = None
        finally:
            if job_id == _extract_job_id:
                _extract_running = False
                _extract_finished_job = job_id

    Thread(target=worker, daemon=True, name=f"tunnel-extract-{job_id}").start()
    return True


def _start_plot_rebuild(
    field_key: str,
    coord_system: str,
    *,
    stage_number: int | None = None,
) -> bool:
    """Rebuild the figure from an in-memory stage bundle (no RS3)."""
    global _extract_running, _extract_started_at, _extract_job_id
    global _extract_error, _pending_figure, _pending_status
    global _pending_stage, _pending_stage_options, _pending_port
    global _pending_bundle, _pending_df
    global _last_field_key, _last_coord, _df, _bundle_df, _active_stage

    if _extract_running:
        _append_log("Job already in progress — ignoring selection change.")
        return False

    stage = int(stage_number) if stage_number is not None else _active_stage
    if stage is None:
        return False
    bundle = _stage_bundles.get(int(stage))
    if bundle is None:
        return False

    _extract_job_id += 1
    job_id = _extract_job_id
    _extract_started_at = time.monotonic()
    _extract_running = True
    _extract_error = None
    _pending_figure = None
    _pending_status = None
    _pending_stage = int(stage)
    _pending_stage_options = []
    _pending_port = None
    _pending_bundle = bundle
    _last_field_key = field_key
    _last_coord = coord_system
    _append_log(
        f"Switching to stage {stage} · {get_field(field_key).label} "
        f"[{effective_coord_for_view(field_key, coord_system)}] from cache…"
    )

    def worker() -> None:
        global _df, _bundle_df, _active_stage
        global _extract_running, _extract_finished_job, _extract_error
        global _pending_figure, _pending_status, _pending_df
        try:
            view = dataframe_from_bundle(bundle, field_key, coord_system)
            fig = build_tunnel_figure(view, progress_callback=_append_log)
            if job_id != _extract_job_id:
                return
            _df = view
            _bundle_df = bundle
            _active_stage = int(stage)
            _pending_df = view
            _pending_figure = fig
            cached_stages = ", ".join(str(s) for s in sorted(_stage_bundles))
            _pending_status = _status_ready(
                len(view),
                field_key,
                coord_system,
                int(stage),
                _last_port,
                from_cache=True,
            ) + f" · stages in memory: {cached_stages}"
            _extract_error = None
        except Exception as exc:  # noqa: BLE001
            if job_id != _extract_job_id:
                return
            msg = str(exc).strip() or exc.__class__.__name__
            _append_log(msg)
            _append_log(traceback.format_exc())
            _extract_error = msg
            _pending_figure = None
            _pending_status = None
            _pending_df = None
        finally:
            if job_id == _extract_job_id:
                _extract_running = False
                _extract_finished_job = job_id

    Thread(target=worker, daemon=True, name=f"tunnel-rebuild-{job_id}").start()
    return True


def _apply_finished_extract(token: int):
    global _extract_applied_job, _pending_figure, _pending_status, _figure
    global _pending_stage, _pending_stage_options, _pending_port
    global _pending_bundle, _pending_df, _bundle_df, _df
    job = _extract_finished_job
    if job <= _extract_applied_job:
        return None
    _extract_applied_job = job

    if _extract_error or _pending_figure is None:
        return (
            no_update,
            "Extract failed — see log.",
            _log_text(),
            token,
            True,
            no_update,
            no_update,
            no_update,
        )

    _figure = _pending_figure
    if _pending_bundle is not None:
        _bundle_df = _pending_bundle
    if _pending_df is not None:
        _df = _pending_df
    status = _pending_status or "Extract complete."
    stage_opts = _pending_stage_options or no_update
    stage_val = _pending_stage if _pending_stage is not None else no_update
    port_val = str(_pending_port) if _pending_port is not None else no_update
    _pending_figure = None
    _pending_status = None
    _pending_stage = None
    _pending_stage_options = []
    _pending_port = None
    _pending_bundle = None
    _pending_df = None
    return (
        _figure,
        status,
        _log_text(),
        token,
        True,
        stage_opts,
        stage_val,
        port_val,
    )


def create_layout(
    *,
    initial_model: str = "",
    field_key: str = "displacement_z",
    coord_system: str = "Global",
    port: int = 60064,
    initial_figure=None,
    status: str = "Load a computed .rs3v3 and click Extract.",
    extract_token: int = 0,
    stage_options: list[dict[str, Any]] | None = None,
    stage_value: int | None = None,
) -> html.Div:
    return html.Div(
        [
            dcc.Interval(id="file-poll", interval=400, n_intervals=0, disabled=True),
            dcc.Interval(id="log-poll", interval=600, n_intervals=0, disabled=True),
            dcc.Store(id="extract-token", data=extract_token),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(
                                "Tunnel Liner Visualization",
                                style={"margin": "0 0 12px 0", "fontSize": "18px"},
                            ),
                            html.Label("Model (.rs3v3)", style={"fontWeight": "bold"}),
                            dcc.Input(
                                id="model-path",
                                type="text",
                                value=initial_model,
                                placeholder="Paste path or Browse…",
                                style=_input(),
                                debounce=True,
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Browse",
                                        id="browse-btn",
                                        n_clicks=0,
                                        style=_btn("#e85a1c"),
                                    ),
                                    html.Button(
                                        "Extract from RS3",
                                        id="extract-btn",
                                        n_clicks=0,
                                        style=_btn("#2b6cb0"),
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "gap": "6px",
                                    "marginBottom": "12px",
                                },
                            ),
                            _labeled(
                                "Port",
                                dcc.Input(
                                    id="port-input",
                                    type="text",
                                    value=str(port),
                                    style=_input(),
                                    debounce=True,
                                ),
                            ),
                            _labeled(
                                "Stage (liner stages)",
                                dcc.Dropdown(
                                    id="stage-dropdown",
                                    options=stage_options or [],
                                    value=stage_value,
                                    placeholder="Extract to list liner stages…",
                                    clearable=False,
                                    searchable=False,
                                ),
                            ),
                            _labeled(
                                "Liner result",
                                dcc.Dropdown(
                                    id="field-dropdown",
                                    options=dropdown_options(),
                                    value=field_key,
                                    clearable=False,
                                    searchable=False,
                                ),
                            ),
                            _labeled(
                                "Coordinate system",
                                dcc.Dropdown(
                                    id="coord-dropdown",
                                    options=[
                                        {"label": "Global", "value": "Global"},
                                        {"label": "Local", "value": "Local"},
                                    ],
                                    value=coord_system,
                                    clearable=False,
                                    searchable=False,
                                ),
                            ),
                            html.Div(
                                id="status",
                                children=status,
                                style={
                                    "fontSize": "13px",
                                    "color": "#2d3748",
                                    "marginTop": "8px",
                                    "marginBottom": "8px",
                                    "minHeight": "36px",
                                },
                            ),
                            html.Label("Log", style={"fontWeight": "bold"}),
                            html.Pre(
                                id="log-box",
                                children=_log_text() or "Ready.",
                                style={
                                    "background": "#1a202c",
                                    "color": "#e2e8f0",
                                    "padding": "10px",
                                    "borderRadius": "4px",
                                    "fontSize": "11px",
                                    "fontFamily": "Consolas, Courier New, monospace",
                                    "height": "220px",
                                    "overflowY": "auto",
                                    "whiteSpace": "pre-wrap",
                                    "margin": 0,
                                },
                            ),
                        ],
                        style={
                            "width": "320px",
                            "minWidth": "280px",
                            "padding": "16px",
                            "boxSizing": "border-box",
                            "borderRight": "1px solid #ddd",
                            "overflowY": "auto",
                            "height": "100vh",
                            "background": "#fafafa",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Graph(
                                id="tunnel-graph",
                                figure=initial_figure or _empty_figure(),
                                style={"height": "100vh"},
                                config={
                                    "scrollZoom": True,
                                    "displaylogo": False,
                                },
                            )
                        ],
                        style={"flex": "1", "height": "100vh"},
                    ),
                ],
                style={"display": "flex", "height": "100vh", "overflow": "hidden"},
            ),
        ]
    )


def create_app(
    *,
    initial_model: str = "",
    field_key: str = "displacement_z",
    coord_system: str = "Global",
    port: int = 60064,
    no_start: bool = False,
    initial_figure=None,
    status: str = "Load a computed .rs3v3 and click Extract.",
    extract_token: int = 0,
    stage_options: list[dict[str, Any]] | None = None,
    stage_value: int | None = None,
) -> Dash:
    global _no_start
    _no_start = no_start

    app = Dash(__name__, suppress_callback_exceptions=True)
    app.title = "Tunnel Liner Visualization"
    app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body { margin: 0; padding: 0; overflow: hidden; height: 100%;
                         font-family: Segoe UI, Arial, sans-serif; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""
    app.layout = create_layout(
        initial_model=initial_model,
        field_key=field_key,
        coord_system=coord_system,
        port=port,
        initial_figure=initial_figure,
        status=status,
        extract_token=extract_token,
        stage_options=stage_options,
        stage_value=stage_value,
    )

    @callback(
        Output("file-poll", "disabled"),
        Input("browse-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def on_browse(_n):
        Thread(
            target=_run_file_dialog,
            kwargs={
                "title": "Select RS3 model",
                "filetypes": [
                    ("RS3 model", "*.rs3v3"),
                    ("All files", "*.*"),
                ],
                "dest": _TEMP_FILE_PATH,
            },
            daemon=True,
        ).start()
        return False

    @callback(
        Output("model-path", "value"),
        Output("file-poll", "disabled", allow_duplicate=True),
        Input("file-poll", "n_intervals"),
        prevent_initial_call=True,
    )
    def poll_model_path(_n):
        try:
            if os.path.isfile(_TEMP_FILE_PATH):
                with open(_TEMP_FILE_PATH, encoding="utf-8") as f:
                    path = f.read().strip()
                os.remove(_TEMP_FILE_PATH)
                if path:
                    return path, True
        except OSError:
            pass
        return no_update, False

    @callback(
        Output("tunnel-graph", "figure"),
        Output("status", "children"),
        Output("log-box", "children"),
        Output("extract-token", "data"),
        Output("log-poll", "disabled"),
        Output("stage-dropdown", "options"),
        Output("stage-dropdown", "value"),
        Output("port-input", "value"),
        Input("extract-btn", "n_clicks"),
        State("model-path", "value"),
        State("port-input", "value"),
        State("field-dropdown", "value"),
        State("coord-dropdown", "value"),
        State("stage-dropdown", "value"),
        State("extract-token", "data"),
        prevent_initial_call=True,
    )
    def on_extract(
        _n, model_path, port_text, field_key_value, coord, stage_value, token
    ):
        try:
            port = int(str(port_text).strip())
        except (TypeError, ValueError):
            return (
                no_update,
                "Invalid RS3 port.",
                _log_text(),
                token,
                True,
                no_update,
                no_update,
                no_update,
            )
        # Fresh Extract with an empty stage list should auto-pick a liner stage.
        # If stages are already listed, honor the user's selection.
        stage_num = None
        if stage_value is not None and _liner_stages:
            stage_num = int(stage_value)
        started = _start_extract_thread(
            str(model_path or "").strip(),
            port,
            field_key_value or "displacement_z",
            coord or CoordSystem.GLOBAL.value,
            stage_number=stage_num,
        )
        if not started:
            return (
                no_update,
                "Extract already running…",
                _log_text(),
                token,
                False,
                no_update,
                no_update,
                no_update,
            )
        return (
            no_update,
            "Extracting from RS3…",
            _log_text(),
            int(token or 0) + 1,
            False,
            no_update,
            no_update,
            no_update,
        )

    @callback(
        Output("coord-dropdown", "value"),
        Output("tunnel-graph", "figure", allow_duplicate=True),
        Output("status", "children", allow_duplicate=True),
        Output("log-box", "children", allow_duplicate=True),
        Output("extract-token", "data", allow_duplicate=True),
        Output("log-poll", "disabled", allow_duplicate=True),
        Output("stage-dropdown", "options", allow_duplicate=True),
        Output("stage-dropdown", "value", allow_duplicate=True),
        Output("port-input", "value", allow_duplicate=True),
        Input("field-dropdown", "value"),
        Input("coord-dropdown", "value"),
        Input("stage-dropdown", "value"),
        State("model-path", "value"),
        State("port-input", "value"),
        State("extract-token", "data"),
        prevent_initial_call=True,
    )
    def on_selection_change(
        field_key_value, coord, stage_value, model_path, port_text, token
    ):
        """Re-plot from cache on field/coord change; re-extract on stage change."""
        from dash import ctx

        if not _liner_stages or _extract_running:
            # Still sync default axes when the field changes, even before Extract.
            if ctx.triggered_id == "field-dropdown":
                try:
                    return (
                        get_field(field_key_value).default_coord.value,
                        *(no_update,) * 8,
                    )
                except KeyError:
                    pass
            return (no_update,) * 9

        field = field_key_value or _last_field_key
        coord_out = no_update
        if ctx.triggered_id == "field-dropdown":
            try:
                coord = get_field(field).default_coord.value
                coord_out = coord
            except KeyError:
                coord = coord or _last_coord
        else:
            coord = coord or _last_coord

        stage_num = None
        if stage_value is not None:
            stage_num = int(stage_value)

        same_selection = (
            field == _last_field_key
            and str(coord) == str(_last_coord)
            and stage_num is not None
            and _active_stage is not None
            and stage_num == int(_active_stage)
        )
        if same_selection:
            return (coord_out, *(no_update,) * 8)

        stage_changed = (
            stage_num is not None
            and _active_stage is not None
            and stage_num != int(_active_stage)
        ) or ctx.triggered_id == "stage-dropdown"

        # Prefer in-memory stage bundles for field / coord / known stage switches.
        if _stage_bundles:
            target_stage = (
                stage_num if stage_num is not None else _active_stage
            )
            if (
                target_stage is not None
                and int(target_stage) in _stage_bundles
                and (
                    ctx.triggered_id in {"field-dropdown", "coord-dropdown"}
                    or stage_changed
                )
            ):
                started = _start_plot_rebuild(
                    field,
                    str(coord),
                    stage_number=int(target_stage),
                )
                if started:
                    try:
                        status = (
                            f"Updating view: stage {target_stage} · "
                            f"{get_field(field).label} "
                            f"[{effective_coord_for_view(field, coord)}]…"
                        )
                    except KeyError:
                        status = "Updating view…"
                    return (
                        coord_out,
                        no_update,
                        status,
                        _log_text(),
                        int(token or 0) + 1,
                        False,
                        no_update,
                        no_update,
                        no_update,
                    )

        try:
            port = int(str(port_text or _last_port).strip())
        except (TypeError, ValueError):
            port = _last_port

        path = str(model_path or _last_model_path).strip()
        if not path:
            return (coord_out, *(no_update,) * 8)

        started = _start_extract_thread(
            path,
            port,
            field,
            coord,
            stage_number=stage_num,
        )
        if not started:
            return (coord_out, *(no_update,) * 8)

        status = (
            f"Loading stage {stage_num}…"
            if stage_num is not None
            else "Extracting…"
        )
        return (
            coord_out,
            no_update,
            status,
            _log_text(),
            int(token or 0) + 1,
            False,
            no_update,
            no_update,
            no_update,
        )

    @callback(
        Output("tunnel-graph", "figure", allow_duplicate=True),
        Output("status", "children", allow_duplicate=True),
        Output("log-box", "children", allow_duplicate=True),
        Output("extract-token", "data", allow_duplicate=True),
        Output("log-poll", "disabled", allow_duplicate=True),
        Output("stage-dropdown", "options", allow_duplicate=True),
        Output("stage-dropdown", "value", allow_duplicate=True),
        Output("port-input", "value", allow_duplicate=True),
        Input("log-poll", "n_intervals"),
        State("extract-token", "data"),
        prevent_initial_call=True,
    )
    def on_log_poll(_n, token):
        applied = _apply_finished_extract(int(token or 0))
        if applied is not None:
            return applied
        if _extract_running:
            return (
                no_update,
                "Working… see log.",
                _log_text(),
                token,
                False,
                no_update,
                no_update,
                no_update,
            )
        return (
            no_update,
            no_update,
            _log_text(),
            token,
            True,
            no_update,
            no_update,
            no_update,
        )

    return app


def run_web_viewer(
    *,
    model: Path | None = None,
    port: int = 60064,
    field_key: str = "displacement_z",
    coord_system: str | None = None,
    no_start: bool = False,
    auto_extract: bool = False,
    http_port: int = _DEFAULT_HTTP_PORT,
) -> int:
    """Start Dash server and open the browser."""
    global _df, _bundle_df, _stage_bundles, _figure, _no_start, _liner_stages
    global _active_stage, _last_port, _last_field_key, _last_coord

    _no_start = no_start
    if coord_system is None:
        try:
            coord_system = get_field(field_key).default_coord.value
        except KeyError:
            coord_system = CoordSystem.GLOBAL.value

    initial = str(model.resolve()) if model is not None else ""
    initial_figure = None
    status = "Load a computed .rs3v3 and click Extract."
    extract_token = 0
    stage_options: list[dict[str, Any]] = []
    stage_value: int | None = None

    if auto_extract and model is not None:
        try:
            _append_log("Auto-extract starting…")
            view, bundle, used_stage, liner_stages, used_port = _extract_from_model(
                str(model.resolve()),
                port,
                field_key,
                coord_system,
                no_start=no_start,
            )
            _df = view
            _bundle_df = bundle
            _stage_bundles = {int(used_stage): bundle}
            _liner_stages = liner_stages
            _active_stage = used_stage
            _last_port = used_port
            _last_field_key = field_key
            _last_coord = coord_system
            port = used_port
            initial_figure = build_tunnel_figure(view, progress_callback=_append_log)
            _figure = initial_figure
            stage_options = _stage_options(liner_stages)
            stage_value = used_stage
            status = _status_ready(
                len(view), field_key, coord_system, used_stage, used_port
            )
            extract_token = 1
            _append_log("Auto-extract done.")
        except Exception:  # noqa: BLE001
            _append_log(traceback.format_exc())
            status = "Auto-extract failed — see log. Fix RS3, then click Extract."

    app = create_app(
        initial_model=initial,
        field_key=field_key,
        coord_system=coord_system,
        port=port,
        no_start=no_start,
        initial_figure=initial_figure,
        status=status,
        extract_token=extract_token,
        stage_options=stage_options,
        stage_value=stage_value,
    )

    def open_browser():
        webbrowser.open(f"http://127.0.0.1:{http_port}")

    Timer(1.2, open_browser).start()
    print("\n" + "=" * 50)
    print("Tunnel Liner Visualization (Web)")
    print("=" * 50)
    print(f"\nStarting server at http://127.0.0.1:{http_port}")
    print("Browser will open automatically...")
    print("Press Ctrl+C to stop.\n")
    app.run(debug=False, host="127.0.0.1", port=http_port, threaded=True)
    return 0
