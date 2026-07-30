"""
Launch the interactive tunnel liner visualization (web / Dash).

Browse a computed .rs3v3, extract liner results via RS3 scripting, and plot
cross-sections — same stack as the FoS contour viewer.

Examples
--------
    python run_tunnel_viewer.py

    python run_tunnel_viewer.py "my_tunnel.rs3v3" --auto

    python run_tunnel_viewer.py --field moment_y --coord Local
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive tunnel liner visualization (web)."
    )
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=None,
        help="Optional path to a computed .rs3v3 model.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=60064,
        help="RS3 scripting server port (default: 60064).",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8052,
        help="Local web server port (default: 8052).",
    )
    parser.add_argument(
        "--field",
        default="displacement_z",
        help=(
            "Liner result field key, e.g. displacement_z, axial_force_x, "
            "moment_y (default: displacement_z)."
        ),
    )
    parser.add_argument(
        "--coord",
        default=None,
        choices=["Global", "Local"],
        help="Result coordinate system (default depends on field).",
    )
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="Do not launch RS3 if no scripting server is running.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="If a model path is given, extract before opening the browser.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = args.model.resolve() if args.model is not None else None
    if model is not None and not model.exists():
        print(f"Model not found: {model}", file=sys.stderr)
        return 1

    from tunnel_viz.rs3_bootstrap import ensure_rs3_protobuf_imports

    ensure_rs3_protobuf_imports()

    from tunnel_viz.web_app import run_web_viewer

    return run_web_viewer(
        model=model,
        port=args.port,
        field_key=args.field,
        coord_system=args.coord,
        no_start=args.no_start,
        auto_extract=bool(args.auto and model is not None),
        http_port=args.http_port,
    )


if __name__ == "__main__":
    raise SystemExit(main())
