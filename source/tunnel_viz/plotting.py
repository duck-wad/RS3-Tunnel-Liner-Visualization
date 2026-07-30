"""Build the interactive 2D/3D tunnel cross-section Plotly figure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _uniform_filter1d(values: np.ndarray, size: int) -> np.ndarray:
    """Moving-average smoother (scipy-compatible enough for tunnel shape)."""
    try:
        from scipy.ndimage import uniform_filter1d

        return uniform_filter1d(values, size=size, mode="nearest")
    except ImportError:
        size = max(1, int(size))
        if size == 1:
            return np.asarray(values, dtype=float)
        kernel = np.ones(size, dtype=float) / size
        pad = size // 2
        padded = np.pad(np.asarray(values, dtype=float), pad, mode="edge")
        return np.convolve(padded, kernel, mode="valid")


def _empty_figure(message: str = "Load a model and click Extract.") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="#666"),
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def dataframe_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df.columns) < 4:
        raise ValueError(
            f"CSV must have at least 4 columns (X, Y, Z, value). Found {len(df.columns)}."
        )
    return df


def _choose_tunnel_axis_index(
    coord1: np.ndarray, coord2: np.ndarray, coord3: np.ndarray
) -> int:
    """
    Pick the tunnel drive axis.

    Early construction stages often have a short lined length, so the cross-section
    width can exceed the drive length. Prefer the axis whose mid-slice looks like a
    closed ring in the other two coordinates, not merely the longest range.
    """
    coords = (coord1, coord2, coord3)
    best_i = int(np.argmax([np.ptp(c) for c in coords]))
    best_score = (-1.0, -1.0)

    for i, axis in enumerate(coords):
        span = float(np.ptp(axis))
        if span <= 0:
            continue
        a = coords[(i + 1) % 3]
        b = coords[(i + 2) % 3]
        mid = float(np.median(axis))
        tol = max(0.25, 0.03 * span)
        mask = np.abs(axis - mid) < tol
        if np.count_nonzero(mask) < 12:
            tol = max(tol, 0.08 * span)
            mask = np.abs(axis - mid) < tol
        if np.count_nonzero(mask) < 12:
            continue

        aa = a[mask]
        bb = b[mask]
        cx = float(np.mean(aa))
        cy = float(np.mean(bb))
        angles = np.arctan2(bb - cy, aa - cx)
        occupied = np.histogram(angles, bins=np.linspace(-np.pi, np.pi, 37))[0]
        coverage = float(np.mean(occupied > 0))
        score = (coverage, span)
        if score > best_score:
            best_score = score
            best_i = i

    return best_i


def build_tunnel_figure(
    df: pd.DataFrame,
    *,
    n_slider_positions: int = 30,
    progress_callback=None,
) -> go.Figure:
    """
    Create the 2D cross-section + 3D tunnel surface figure from point data.

    Expects at least four columns: three coordinates and one value column.
    The tunnel axis is chosen as the drive direction (ring-like cross-sections),
    not merely the coordinate with the largest range.
    """
    progress = progress_callback or (lambda _msg: None)

    if df is None or len(df) == 0:
        return _empty_figure("No data points to plot.")
    if len(df.columns) < 4:
        raise ValueError("Data must have at least 4 columns (X, Y, Z, value).")

    coord1 = df.iloc[:, 0].to_numpy(dtype=float)
    coord2 = df.iloc[:, 1].to_numpy(dtype=float)
    coord3 = df.iloc[:, 2].to_numpy(dtype=float)
    values = df.iloc[:, 3].to_numpy(dtype=float)

    coord1_name = str(df.columns[0])
    coord2_name = str(df.columns[1])
    coord3_name = str(df.columns[2])
    values_column_name = str(df.columns[3])

    ranges = [
        float(np.nanmax(coord1) - np.nanmin(coord1)),
        float(np.nanmax(coord2) - np.nanmin(coord2)),
        float(np.nanmax(coord3) - np.nanmin(coord3)),
    ]
    tunnel_axis_idx = _choose_tunnel_axis_index(coord1, coord2, coord3)

    if tunnel_axis_idx == 0:
        y, y_name = coord1, coord1_name
        x, x_name = coord2, coord2_name
        z, z_name = coord3, coord3_name
    elif tunnel_axis_idx == 1:
        y, y_name = coord2, coord2_name
        x, x_name = coord1, coord1_name
        z, z_name = coord3, coord3_name
    else:
        y, y_name = coord3, coord3_name
        x, x_name = coord1, coord1_name
        z, z_name = coord2, coord2_name

    progress(
        f"Tunnel axis = {y_name} "
        f"(ranges X={ranges[0]:.3g}, Y={ranges[1]:.3g}, Z={ranges[2]:.3g})"
    )

    global_center_x = float(np.mean(x))
    global_center_z = float(np.mean(z))
    all_radii = np.sqrt((x - global_center_x) ** 2 + (z - global_center_z) ** 2)
    global_tunnel_radius = float(np.mean(all_radii))

    progress("Calculating average tunnel cross-section shape…")
    all_angles = np.arctan2(z - global_center_z, x - global_center_x)
    num_angle_bins = 360
    angle_bins = np.linspace(-np.pi, np.pi, num_angle_bins + 1)
    bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2

    avg_shape_radii = np.zeros(num_angle_bins)
    has_data = np.zeros(num_angle_bins, dtype=bool)
    for i in range(num_angle_bins):
        in_bin = (all_angles >= angle_bins[i]) & (all_angles < angle_bins[i + 1])
        if np.sum(in_bin) > 0:
            avg_shape_radii[i] = np.mean(all_radii[in_bin])
            has_data[i] = True

    if not np.all(has_data):
        valid_indices = np.where(has_data)[0]
        if len(valid_indices) == 0:
            raise RuntimeError("Could not determine tunnel shape from the data.")
        avg_shape_radii = np.interp(
            bin_centers,
            bin_centers[valid_indices],
            avg_shape_radii[valid_indices],
            period=2 * np.pi,
        )

    padded_radii = np.concatenate(
        [avg_shape_radii[-30:], avg_shape_radii, avg_shape_radii[:30]]
    )
    smoothed_padded = _uniform_filter1d(padded_radii, size=25)
    avg_shape_radii = smoothed_padded[30:-30]
    global_tunnel_shape_angles = bin_centers
    global_tunnel_shape_radii = avg_shape_radii

    y_min, y_max = float(np.min(y)), float(np.max(y))
    slider_positions = np.linspace(y_min, y_max, n_slider_positions)
    init_idx = len(slider_positions) // 2

    def get_cross_section(y_pos, tolerance=0.2, num_points=50):
        mask = np.abs(y - y_pos) < tolerance
        section_x = x[mask]
        section_z = z[mask]
        section_vals = values[mask]
        if len(section_x) < 4:
            return None

        center_x = global_center_x
        center_z = global_center_z
        angles = np.arctan2(section_z - center_z, section_x - center_x)

        # Angular bin average avoids zigzags from duplicate/near-duplicate angles.
        # endpoint=False so -π and +π are not both sampled (same physical seam).
        theta_open = np.linspace(-np.pi, np.pi, num_points, endpoint=False)
        bin_half = np.pi / num_points
        vals_open = np.full(num_points, np.nan)
        for i, t in enumerate(theta_open):
            d = np.abs(np.angle(np.exp(1j * (angles - t))))
            in_bin = d < bin_half * 1.5
            if np.any(in_bin):
                vals_open[i] = float(np.mean(section_vals[in_bin]))

        valid = np.isfinite(vals_open)
        if np.count_nonzero(valid) < 4:
            return None
        if not np.all(valid):
            vals_open = np.interp(
                theta_open,
                theta_open[valid],
                vals_open[valid],
                period=2 * np.pi,
            )

        tunnel_shape_radii = np.interp(
            theta_open,
            global_tunnel_shape_angles,
            global_tunnel_shape_radii,
            period=2 * np.pi,
        )

        max_abs_val = max(abs(float(np.min(vals_open))), abs(float(np.max(vals_open))))
        if max_abs_val == 0:
            max_abs_val = 1.0
        scale_factor = global_tunnel_radius * 0.3

        tunnel_x = center_x + tunnel_shape_radii * np.cos(theta_open)
        tunnel_z = center_z + tunnel_shape_radii * np.sin(theta_open)
        value_offset = (vals_open / max_abs_val) * scale_factor
        value_radius = tunnel_shape_radii + value_offset
        value_x = center_x + value_radius * np.cos(theta_open)
        value_z = center_z + value_radius * np.sin(theta_open)

        # Close the loop for Plotly (first point repeated).
        tunnel_x = np.append(tunnel_x, tunnel_x[0])
        tunnel_z = np.append(tunnel_z, tunnel_z[0])
        value_x = np.append(value_x, value_x[0])
        value_z = np.append(value_z, value_z[0])
        vals_closed = np.append(vals_open, vals_open[0])

        max_idx = int(np.argmax(vals_open))
        min_idx = int(np.argmin(vals_open))
        return {
            "tunnel_x": tunnel_x,
            "tunnel_z": tunnel_z,
            "value_x": value_x,
            "value_z": value_z,
            "values": vals_closed,
            "max_x": value_x[max_idx],
            "max_z": value_z[max_idx],
            "max_value": float(vals_open[max_idx]),
            "min_x": value_x[min_idx],
            "min_z": value_z[min_idx],
            "min_value": float(vals_open[min_idx]),
        }

    fixed_window_size = global_tunnel_radius * 1.6
    x_axis_min = -fixed_window_size
    x_axis_max = fixed_window_size
    z_axis_min = -fixed_window_size
    z_axis_max = fixed_window_size

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scatter"}, {"type": "scatter3d"}]],
        column_widths=[0.4, 0.6],
        horizontal_spacing=0.05,
        subplot_titles=("2D Cross-Section (X-Z Plane)", "3D Tunnel View"),
    )

    init_y_pos = float(slider_positions[init_idx])
    section_data = get_cross_section(init_y_pos)
    if section_data is None:
        # Widen tolerance until we find a usable section.
        for tol in (0.5, 1.0, 2.0, 5.0):
            section_data = get_cross_section(init_y_pos, tolerance=tol)
            if section_data is not None:
                break
    if section_data is None:
        raise RuntimeError("Could not generate an initial cross-section from the data.")

    fig.add_trace(
        go.Scatter(
            x=section_data["tunnel_x"],
            y=section_data["tunnel_z"],
            mode="lines",
            line=dict(color="black", width=2),
            name="Tunnel",
            hoverinfo="skip",
            fill="toself",
            fillcolor="lightgray",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=section_data["value_x"],
            y=section_data["value_z"],
            mode="lines",
            line=dict(color="orange", width=3),
            name=values_column_name,
            hovertemplate=(
                f"{x_name}: %{{customdata[0]:.2f}}<br>"
                f"{z_name}: %{{customdata[1]:.2f}}<br>"
                f"{values_column_name}: %{{customdata[2]:.3f}}<extra></extra>"
            ),
            customdata=np.column_stack(
                [
                    section_data["value_x"],
                    section_data["value_z"],
                    section_data["values"],
                ]
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[section_data["max_x"]],
            y=[section_data["max_z"]],
            mode="markers+text",
            marker=dict(size=10, color="red", symbol="circle"),
            text=[f"Max: {section_data['max_value']:.3f}"],
            textposition="top center",
            textfont=dict(size=10, color="red"),
            name="Maximum",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[section_data["min_x"]],
            y=[section_data["min_z"]],
            mode="markers+text",
            marker=dict(size=10, color="blue", symbol="circle"),
            text=[f"Min: {section_data['min_value']:.3f}"],
            textposition="bottom center",
            textfont=dict(size=10, color="blue"),
            name="Minimum",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    progress("Building 3D tunnel surface…")
    num_theta = 120
    num_y_sections = 100
    y_range = y_max - y_min
    y_trim = y_range * 0.02
    y_mesh_min = y_min + y_trim
    y_mesh_max = y_max - y_trim
    # endpoint=False avoids duplicate seam columns at 0 and 2π.
    theta_mesh = np.linspace(0, 2 * np.pi, num_theta, endpoint=False)
    y_mesh = np.linspace(y_mesh_min, y_mesh_max, num_y_sections)
    theta_grid, y_grid = np.meshgrid(theta_mesh, y_mesh)

    theta_mesh_shifted = np.where(theta_mesh > np.pi, theta_mesh - 2 * np.pi, theta_mesh)
    radii_at_angles = np.interp(
        theta_mesh_shifted,
        global_tunnel_shape_angles,
        global_tunnel_shape_radii,
        period=2 * np.pi,
    )
    radii_grid = np.tile(radii_at_angles, (num_y_sections, 1))
    x_surf = global_center_x + radii_grid * np.cos(theta_grid)
    z_surf = global_center_z + radii_grid * np.sin(theta_grid)

    surf_values = np.full_like(y_grid, np.nan, dtype=float)
    y_tolerance = 0.25
    angle_tolerance = 0.4
    for i in range(num_y_sections):
        y_val = y_mesh[i]
        mask = np.abs(y - y_val) < y_tolerance
        if not np.any(mask):
            mask = np.abs(y - y_val) < y_tolerance * 2
        if not np.any(mask):
            continue
        point_angles = np.arctan2(z[mask] - global_center_z, x[mask] - global_center_x)
        point_vals = values[mask]
        for j in range(num_theta):
            # Compare in a common angle space (arctan2 range).
            angle = theta_mesh_shifted[j]
            angle_diff = np.abs(np.angle(np.exp(1j * (point_angles - angle))))
            close_angles = angle_diff < angle_tolerance
            if np.any(close_angles):
                surf_values[i, j] = float(np.mean(point_vals[close_angles]))

    # Fill any remaining gaps along theta, then along Y.
    for i in range(num_y_sections):
        row = surf_values[i, :]
        valid = np.isfinite(row)
        if np.any(valid) and not np.all(valid):
            surf_values[i, :] = np.interp(
                theta_mesh_shifted,
                theta_mesh_shifted[valid],
                row[valid],
                period=2 * np.pi,
            )
    for j in range(num_theta):
        col = surf_values[:, j]
        valid = np.isfinite(col)
        if np.any(valid) and not np.all(valid):
            surf_values[:, j] = np.interp(y_mesh, y_mesh[valid], col[valid])

    # Close the surface visually by repeating the first theta column.
    x_surf = np.concatenate([x_surf, x_surf[:, :1]], axis=1)
    y_grid = np.concatenate([y_grid, y_grid[:, :1]], axis=1)
    z_surf = np.concatenate([z_surf, z_surf[:, :1]], axis=1)
    surf_values = np.concatenate([surf_values, surf_values[:, :1]], axis=1)

    hover_text = np.empty_like(surf_values, dtype=object)
    n_theta_closed = surf_values.shape[1]
    for i in range(num_y_sections):
        for j in range(n_theta_closed):
            hover_text[i, j] = (
                f"{x_name}: {x_surf[i, j]:.2f}<br>"
                f"{y_name}: {y_grid[i, j]:.2f}<br>"
                f"{z_name}: {z_surf[i, j]:.2f}<br>"
                f"{values_column_name}: {surf_values[i, j]:.3f}"
            )

    fig.add_trace(
        go.Surface(
            x=x_surf,
            y=y_grid,
            z=z_surf,
            surfacecolor=surf_values,
            colorscale="rainbow",
            showscale=True,
            colorbar=dict(
                title=values_column_name,
                x=1.05,
                y=0.5,
                yanchor="middle",
                len=0.9,
            ),
            cmin=float(np.nanmin(values)),
            cmax=float(np.nanmax(values)),
            name="Tunnel",
            text=hover_text,
            hoverinfo="text",
        ),
        row=1,
        col=2,
    )

    plane_size = global_tunnel_radius * 1.5
    px0 = global_center_x - plane_size
    px1 = global_center_x + plane_size
    pz0 = global_center_z - plane_size
    pz1 = global_center_z + plane_size
    # 4-vertex Mesh3d plane (cheap to restyle; full Surface was too heavy).
    plane_x = np.array([px0, px1, px1, px0], dtype=float)
    plane_z = np.array([pz0, pz0, pz1, pz1], dtype=float)
    plane_y0 = np.full(4, init_y_pos, dtype=float)
    fig.add_trace(
        go.Mesh3d(
            x=plane_x,
            y=plane_y0,
            z=plane_z,
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color="red",
            opacity=0.35,
            name=f"Slice at {y_name}={init_y_pos:.1f}",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        scene=dict(
            xaxis_title=f"{x_name} Position",
            yaxis_title=f"{y_name} (Tunnel Axis)",
            zaxis_title=f"{z_name} Position",
            aspectratio=dict(x=1, y=3, z=1),
            camera=dict(eye=dict(x=1.5, y=-2, z=1.2)),
            domain=dict(x=[0.45, 0.99], y=[0.05, 0.95]),
        ),
        margin=dict(l=0, r=0, b=100, t=30),
        hoverlabel=dict(
            bgcolor="rgb(68, 68, 68)",
            font_size=13,
            font_family="Arial",
            font_color="white",
        ),
        legend=dict(
            x=0.02,
            y=0.92,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        title_text=f"{x_name} Position",
        row=1,
        col=1,
        scaleanchor="y",
        scaleratio=1,
        range=[global_center_x + x_axis_min, global_center_x + x_axis_max],
        domain=[0.0, 0.4],
    )
    fig.update_yaxes(
        title_text=f"{z_name} Position",
        row=1,
        col=1,
        range=[global_center_z + z_axis_min, global_center_z + z_axis_max],
        domain=[0.05, 0.95],
    )

    # Restyle (not animate/redraw=False): 3D slice must actually move.
    # Use plain lists so Dash JSON-serializes slider steps reliably.
    plane_x_list = plane_x.tolist()
    plane_z_list = plane_z.tolist()
    steps = []
    for y_pos in slider_positions:
        section = get_cross_section(float(y_pos), num_points=36)
        if section is None:
            continue
        y_val = float(y_pos)
        steps.append(
            dict(
                method="restyle",
                args=[
                    {
                        "x": [
                            np.asarray(section["tunnel_x"]).tolist(),
                            np.asarray(section["value_x"]).tolist(),
                            [float(section["max_x"])],
                            [float(section["min_x"])],
                            plane_x_list,
                        ],
                        "y": [
                            np.asarray(section["tunnel_z"]).tolist(),
                            np.asarray(section["value_z"]).tolist(),
                            [float(section["max_z"])],
                            [float(section["min_z"])],
                            [y_val, y_val, y_val, y_val],
                        ],
                        "z": [None, None, None, None, plane_z_list],
                        "text": [
                            None,
                            None,
                            [f"Max: {section['max_value']:.3f}"],
                            [f"Min: {section['min_value']:.3f}"],
                            None,
                        ],
                    },
                    [0, 1, 2, 3, 5],
                ],
                label=f"{y_val:.1f}",
            )
        )

    fig.update_layout(
        sliders=[
            dict(
                active=min(init_idx, max(len(steps) - 1, 0)) if steps else 0,
                currentvalue={"prefix": f"Tunnel Position {y_name} = "},
                pad={"t": 50},
                steps=steps,
                x=0.1,
                y=0.05,
                len=0.8,
                transition={"duration": 0},
            )
        ]
    )
    progress(f"Figure ready ({len(df):,} points, {len(steps)} slider steps).")
    return fig
