"""Extract tunnel liner nodal results from an RS3 model via scripting."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd

from .result_types import (
    LINER_FIELDS,
    CoordSystem,
    LinerField,
    cache_column_name,
    get_field,
    is_global_mesh_field,
)

if TYPE_CHECKING:
    from rs3.Model import Model


@dataclass(frozen=True)
class StageInfo:
    """One analysis stage that can appear in the stage dropdown."""

    number: int
    name: str
    has_liner: bool


def _call_with_timeout(fn: Callable, timeout_s: float | None, label: str):
    if timeout_s is None or timeout_s <= 0:
        return fn()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeout as exc:
            raise TimeoutError(
                f"Timed out after {timeout_s:g}s while {label}."
            ) from exc


def list_model_stages(
    model: "Model",
    *,
    progress_callback: Any | None = None,
) -> list[StageInfo]:
    """Return all stages with names and whether liner properties are in use."""
    progress = progress_callback or (lambda _msg: None)
    stages_api = model.ProjectSettings.Stages
    names = list(stages_api.getDefinedStageNames())
    if not names:
        # Fallback: probe until setActiveStage / property query fails.
        names = [f"Stage {i}" for i in range(1, 64)]
        probed: list[StageInfo] = []
        for i, name in enumerate(names, start=1):
            try:
                has = _stage_has_liner_properties(model, i)
            except Exception:  # noqa: BLE001
                break
            probed.append(StageInfo(number=i, name=name, has_liner=has))
        return probed

    out: list[StageInfo] = []
    for i, name in enumerate(names, start=1):
        label = str(name).strip() or f"Stage {i}"
        try:
            has = _stage_has_liner_properties(model, i)
        except Exception as exc:  # noqa: BLE001
            progress(f"Stage {i} ({label}): could not check liners ({exc})")
            has = False
        out.append(StageInfo(number=i, name=label, has_liner=has))
        progress(
            f"Stage {i}: {label} — "
            f"{'liner present' if has else 'no liner'}"
        )
    return out


def list_liner_stages(
    model: "Model",
    *,
    progress_callback: Any | None = None,
) -> list[StageInfo]:
    """Stages that have liner / lining-composition properties assigned."""
    stages = list_model_stages(model, progress_callback=progress_callback)
    liner = [s for s in stages if s.has_liner]
    progress = progress_callback or (lambda _msg: None)
    if liner:
        progress(
            f"Found {len(liner)} stage(s) with liner: "
            + ", ".join(f"{s.number} ({s.name})" for s in liner)
        )
    else:
        progress("No stages report liner / lining-composition properties in use.")
    return liner


def _stage_has_liner_properties(model: "Model", stage_number: int) -> bool:
    """True if simple liner or lining composition is assigned at this stage."""
    try:
        liners = model.getLinerPropertiesInUseByStage(stage_number) or []
    except Exception:  # noqa: BLE001
        liners = []
    try:
        comps = model.getLiningCompositionPropertiesInUseByStage(stage_number) or []
    except Exception:  # noqa: BLE001
        comps = []
    return bool(liners) or bool(comps)


def resolve_stage_number(
    model: "Model",
    stage_number: int | None = None,
    *,
    liner_stages: list[StageInfo] | None = None,
) -> int:
    """
    Pick a stage to extract.

    Preference: explicit stage (if it has a liner when ``liner_stages`` is
    provided) → active stage if it has a liner → last liner stage → active → 1.
    """
    liner_nums = {s.number for s in liner_stages} if liner_stages is not None else None

    if stage_number is not None and stage_number >= 1:
        if liner_nums is None or int(stage_number) in liner_nums:
            return int(stage_number)

    active: int | None = None
    try:
        active = int(model.getActiveStage())
        if active < 1:
            active = None
    except Exception:  # noqa: BLE001
        active = None

    if active is not None and (liner_nums is None or active in liner_nums):
        return active
    if liner_stages:
        return liner_stages[-1].number
    if active is not None:
        return active
    return 1


def _to_float_list(raw: Any) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, (float, int, np.floating, np.integer)):
        return [float(raw)]
    try:
        return [float(v) for v in list(raw)]
    except TypeError:
        return [float(raw)]


# Liner displacement fields that have a matching global solid-mesh quantity.
_GLOBAL_DISPLACEMENT_ENUMS = {
    "DISPLACEMENT_X": "DISPLACEMENT_X",
    "DISPLACEMENT_Y": "DISPLACEMENT_Y",
    "DISPLACEMENT_Z": "DISPLACEMENT_Z",
    "TOTAL_DISPLACEMENT": "TOTAL_DISPLACEMENT",
}


def _element_node_values(el, data_type) -> list[float]:
    """
    One result value per attached node.

    In practice RS3 stores one ``LinerResults`` message whose fields are length
    ``n_nodes`` (same as ``getResults()``). Prefer that layout. Fall back to
    one message per node when needed.
    """
    attached = list(el.AttachedNodeIDs)
    n = len(attached)
    if n == 0:
        return []

    try:
        nodal = list(el._linerResults)
    except Exception:  # noqa: BLE001
        try:
            nodal = list(el._grpcElementResults.linerNodalResults)
        except Exception:  # noqa: BLE001
            nodal = []

    if not nodal:
        return []

    # Usual layout: first (often only) message holds one value per node.
    fl0 = _to_float_list(getattr(nodal[0], data_type.value, None))
    if len(fl0) == n:
        return fl0

    # One scalar (or length-1 field) per attached node.
    if len(nodal) == n:
        out: list[float] = []
        for i, entry in enumerate(nodal):
            fl = _to_float_list(getattr(entry, data_type.value, None))
            if not fl:
                out.append(float("nan"))
            elif len(fl) == 1:
                out.append(float(fl[0]))
            elif len(fl) == n:
                out.append(float(fl[i]))
            else:
                out.append(float(fl[0]))
        return out

    if len(fl0) == 1:
        return [fl0[0]] * n

    combined: list[float] = []
    for entry in nodal:
        combined.extend(_to_float_list(getattr(entry, data_type.value, None)))
    if len(combined) == n:
        return combined
    return []


def _set_liner_coordinate_system(liner, coord: CoordSystem, progress: Callable) -> str:
    """
    Ask RS3 for Global/Local axes. Many models only expose named frames
    (e.g. ``Tunnel Frames 1``); ``Global`` / ``Local`` then fail.
    """
    try:
        current = str(liner.getCoordinateSystem())
    except Exception:  # noqa: BLE001
        current = ""

    if coord is CoordSystem.GLOBAL:
        candidates = ["Global", "global", "GLOBAL", "World", "XYZ"]
    else:
        candidates = ["Local", "local", "LOCAL", "Element Local"]
        if current:
            candidates.insert(0, current)

    last_err: Exception | None = None
    for name in candidates:
        try:
            liner.setCoordinateSystem(name)
            got = str(liner.getCoordinateSystem())
            progress(f"Coordinate system set to {got!r}.")
            return got
        except Exception as exc:  # noqa: BLE001
            last_err = exc

    progress(
        f"Could not set coordinate system to {coord.value} "
        f"(current={current!r}, last error={last_err}). "
        "Continuing with RS3's current axes."
    )
    return current


def extract_liner_dataframe(
    model: "Model",
    *,
    field: str | LinerField = "displacement_z",
    stage_number: int | None = None,
    srf_result_index: int = 0,
    coordinate_system: str | CoordSystem | None = None,
    entity_name: str | None = None,
    unique_nodes: bool = True,
    rpc_timeout_s: float | None = 300.0,
    progress_callback: Any | None = None,
    auto_detect_liner_stage: bool = True,
) -> tuple[pd.DataFrame, int, list[StageInfo]]:
    """
    Pull liner element nodal results and return ``(df, stage, liner_stages)``.

    When ``auto_detect_liner_stage`` is True, stages without liner properties
    are skipped when resolving the stage to extract.
    """
    from rs3.results.ResultEnums import LinerResultTypes

    field_info = get_field(field) if isinstance(field, str) else field
    data_type = LinerResultTypes[field_info.enum_name]

    if coordinate_system is None:
        coord = field_info.default_coord
    elif isinstance(coordinate_system, CoordSystem):
        coord = coordinate_system
    else:
        coord = CoordSystem(str(coordinate_system))

    progress = progress_callback or (lambda _msg: None)

    liner_stages: list[StageInfo] = []
    if auto_detect_liner_stage:
        progress("Detecting stages with liner properties…")
        liner_stages = list_liner_stages(model, progress_callback=progress)
        if not liner_stages:
            # Fall back to all stages so the dropdown still has options; extract
            # may still succeed if results exist without property-in-use flags.
            all_stages = list_model_stages(model, progress_callback=progress)
            liner_stages = all_stages
            progress(
                "No liner-property stages found; falling back to all stages "
                "and probing results."
            )

    stage = resolve_stage_number(
        model,
        stage_number,
        liner_stages=liner_stages if auto_detect_liner_stage else None,
    )

    use_global_mesh = (
        coord is CoordSystem.GLOBAL
        and field_info.enum_name in _GLOBAL_DISPLACEMENT_ENUMS
    )
    if use_global_mesh:
        progress(
            f"Querying global solid-mesh displacements on liner nodes "
            f"(stage={stage}, srfIndex={srf_result_index}, field={field_info.label})…"
        )
    else:
        progress(
            f"Querying composite liner results "
            f"(stage={stage}, srfIndex={srf_result_index}, field={field_info.label}, "
            f"coords={coord.value})…"
        )

    # Only auto-fall through other liner stages when the caller did not pick one.
    explicit_stage = stage_number is not None and stage_number >= 1
    candidates = [stage]
    if auto_detect_liner_stage and not explicit_stage:
        for info in liner_stages:
            if info.number not in candidates:
                candidates.append(info.number)

    last_error: Exception | None = None
    df: pd.DataFrame | None = None
    used_stage = stage

    for candidate in candidates:
        try:
            if use_global_mesh:
                df = _extract_global_displacement_on_liner(
                    model,
                    stage=candidate,
                    field_enum_name=field_info.enum_name,
                    field_label=field_info.label,
                    srf_result_index=srf_result_index,
                    entity_name=entity_name,
                    rpc_timeout_s=rpc_timeout_s,
                    progress=progress,
                )
            else:
                df = _extract_stage(
                    model,
                    stage=candidate,
                    data_type=data_type,
                    field_label=field_info.label,
                    coord=coord,
                    srf_result_index=srf_result_index,
                    entity_name=entity_name,
                    unique_nodes=unique_nodes,
                    rpc_timeout_s=rpc_timeout_s,
                    progress=progress,
                )
            used_stage = candidate
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            progress(f"Stage {candidate}: {exc}")
            if explicit_stage:
                break
            continue

    if df is None:
        raise RuntimeError(
            "Could not extract liner results"
            + (f" for stage {stage}." if explicit_stage else " from any liner stage.")
            + f"\nLast error: {last_error}"
        )

    if used_stage != stage:
        progress(
            f"Auto-selected stage {stage} had no liner mesh; "
            f"using stage {used_stage} instead."
        )

    return df, used_stage, liner_stages


def extract_all_liner_results(
    model: "Model",
    *,
    stage_number: int | None = None,
    srf_result_index: int = 0,
    entity_name: str | None = None,
    rpc_timeout_s: float | None = 300.0,
    progress_callback: Any | None = None,
    auto_detect_liner_stage: bool = True,
) -> tuple[pd.DataFrame, int, list[StageInfo]]:
    """
    Load every supported liner field for one stage into a single DataFrame.

    Columns: ``X, Y, Z`` plus ``{label} [Global]`` / ``{label} [Local]`` as
    available. Global displacements come from the solid mesh; Local quantities
    (displacements, forces, moments) come from one liner element stream.
    """
    progress = progress_callback or (lambda _msg: None)

    liner_stages: list[StageInfo] = []
    if auto_detect_liner_stage:
        progress("Detecting stages with liner properties…")
        liner_stages = list_liner_stages(model, progress_callback=progress)
        if not liner_stages:
            liner_stages = list_model_stages(model, progress_callback=progress)
            progress(
                "No liner-property stages found; falling back to all stages "
                "and probing results."
            )

    stage = resolve_stage_number(
        model,
        stage_number,
        liner_stages=liner_stages if auto_detect_liner_stage else None,
    )
    progress(
        f"Bulk-extracting all liner fields "
        f"(stage={stage}, srfIndex={srf_result_index})…"
    )

    explicit_stage = stage_number is not None and stage_number >= 1
    candidates = [stage]
    if auto_detect_liner_stage and not explicit_stage:
        for info in liner_stages:
            if info.number not in candidates:
                candidates.append(info.number)

    last_error: Exception | None = None
    bundle: pd.DataFrame | None = None
    used_stage = stage

    for candidate in candidates:
        try:
            bundle = _extract_all_for_stage(
                model,
                stage=candidate,
                srf_result_index=srf_result_index,
                entity_name=entity_name,
                rpc_timeout_s=rpc_timeout_s,
                progress=progress,
            )
            used_stage = candidate
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            progress(f"Stage {candidate}: {exc}")
            if explicit_stage:
                break
            continue

    if bundle is None:
        raise RuntimeError(
            "Could not bulk-extract liner results"
            + (f" for stage {stage}." if explicit_stage else " from any liner stage.")
            + f"\nLast error: {last_error}"
        )

    if used_stage != stage:
        progress(
            f"Auto-selected stage {stage} had no liner mesh; "
            f"using stage {used_stage} instead."
        )

    value_cols = [c for c in bundle.columns if c not in {"X", "Y", "Z"}]
    progress(
        f"Cached {len(bundle):,} liner nodes × {len(value_cols)} result columns "
        f"for stage {used_stage}."
    )
    return bundle, used_stage, liner_stages


def _extract_all_for_stage(
    model: "Model",
    *,
    stage: int,
    srf_result_index: int,
    entity_name: str | None,
    rpc_timeout_s: float | None,
    progress: Callable,
) -> pd.DataFrame:
    from rs3.results.ResultEnums import LinerResultTypes, SolidsDataType

    results_api = model.Results
    available = _call_with_timeout(
        lambda: results_api.getResultsAvailability(
            stageNumber=stage,
            srfResultIndex=srf_result_index,
        ),
        rpc_timeout_s,
        f"getResultsAvailability(stage={stage})",
    )
    if not available:
        raise RuntimeError(f"No computed results for stage {stage}.")

    liner_list = _call_with_timeout(
        lambda: results_api.getCompositeLinerResults(
            stageNumber=[stage],
            srfResultIndex=srf_result_index,
        ),
        rpc_timeout_s,
        f"getCompositeLinerResults(stage={stage})",
    )
    if not liner_list:
        raise RuntimeError(f"getCompositeLinerResults returned no handles for stage {stage}.")
    liner = liner_list[0]

    progress(f"Stage {stage}: streaming liner node coordinates…")
    nodes = _call_with_timeout(
        lambda: liner.getLinerNodeResults(entityName=entity_name),
        rpc_timeout_s,
        f"getLinerNodeResults(stage={stage})",
    )
    if not nodes:
        raise RuntimeError(f"No liner nodes at stage {stage}.")

    node_xyz, primary_layer = _primary_layer_node_xyz(
        nodes, progress=progress, stage=stage
    )
    if not node_xyz:
        raise RuntimeError(f"No liner nodes on the primary layer at stage {stage}.")
    progress(f"Stage {stage}: collected coordinates for {len(node_xyz):,} liner nodes.")

    node_ids = list(node_xyz.keys())
    xs = [node_xyz[n][0] for n in node_ids]
    ys = [node_xyz[n][1] for n in node_ids]
    zs = [node_xyz[n][2] for n in node_ids]
    bundle = pd.DataFrame({"X": xs, "Y": ys, "Z": zs}, index=node_ids)

    # --- Global displacements from solid mesh (one query for all) ---
    global_fields = [f for f in LINER_FIELDS if is_global_mesh_field(f.key)]
    solid_types = {
        SolidsDataType[_GLOBAL_DISPLACEMENT_ENUMS[f.enum_name]] for f in global_fields
    }
    progress(
        f"Stage {stage}: streaming global solid-mesh displacements "
        f"({', '.join(f.label for f in global_fields)})…"
    )
    mesh_list = _call_with_timeout(
        lambda: results_api.getMeshResults(
            stageNumber=[stage],
            srfResultIndex=srf_result_index,
            requiredDataTypes=solid_types,
        ),
        rpc_timeout_s,
        f"getMeshResults(stage={stage})",
    )
    if not mesh_list:
        raise RuntimeError(f"getMeshResults returned no handles for stage {stage}.")

    solid_nodes = _call_with_timeout(
        lambda: mesh_list[0].getMeshNodeResults(),
        rpc_timeout_s,
        f"getMeshNodeResults(stage={stage})",
    )
    if not solid_nodes:
        raise RuntimeError(f"No solid mesh nodes at stage {stage}.")

    global_vals: dict[str, dict[Any, float]] = {
        cache_column_name(f.key, CoordSystem.GLOBAL): {} for f in global_fields
    }
    solid_by_field = {
        cache_column_name(f.key, CoordSystem.GLOBAL): SolidsDataType[
            _GLOBAL_DISPLACEMENT_ENUMS[f.enum_name]
        ]
        for f in global_fields
    }
    for snode in solid_nodes:
        nid = snode.NodeID
        if nid not in node_xyz:
            continue
        for col, solid_type in solid_by_field.items():
            try:
                global_vals[col][nid] = float(snode.getResult(solid_type))
            except Exception:  # noqa: BLE001
                continue

    for col, by_id in global_vals.items():
        bundle[col] = [by_id.get(nid, float("nan")) for nid in node_ids]
        n_ok = int(np.sum(np.isfinite(bundle[col].to_numpy(dtype=float))))
        progress(f"Stage {stage}: cached {col} ({n_ok:,}/{len(node_ids):,} nodes).")

    # --- Local / frame liner fields from one element stream ---
    _set_liner_coordinate_system(liner, CoordSystem.LOCAL, progress)
    progress(f"Stage {stage}: streaming liner element results (all Local fields)…")
    elements = _call_with_timeout(
        lambda: liner.getLinerElementResults(entityName=entity_name),
        rpc_timeout_s,
        f"getLinerElementResults(stage={stage})",
    )
    if not elements:
        raise RuntimeError(f"No liner elements with results at stage {stage}.")

    local_fields = list(LINER_FIELDS)
    data_types = {f.key: LinerResultTypes[f.enum_name] for f in local_fields}
    sum_by: dict[str, dict[Any, float]] = {
        cache_column_name(f.key, CoordSystem.LOCAL): {} for f in local_fields
    }
    count_by: dict[str, dict[Any, int]] = {
        cache_column_name(f.key, CoordSystem.LOCAL): {} for f in local_fields
    }

    skipped = 0
    used_elements = 0
    for el in elements:
        try:
            layer = int(el.LayerIndex)
        except Exception:  # noqa: BLE001
            layer = primary_layer
        if layer != primary_layer:
            skipped += 1
            continue

        attached = list(el.AttachedNodeIDs)
        if not attached:
            skipped += 1
            continue

        used_elements += 1
        for field in local_fields:
            col = cache_column_name(field.key, CoordSystem.LOCAL)
            raw_vals = _element_node_values(el, data_types[field.key])
            if len(raw_vals) != len(attached):
                continue
            for nid, value in zip(attached, raw_vals):
                if nid not in node_xyz or value != value:
                    continue
                sum_by[col][nid] = sum_by[col].get(nid, 0.0) + float(value)
                count_by[col][nid] = count_by[col].get(nid, 0) + 1

    for field in local_fields:
        col = cache_column_name(field.key, CoordSystem.LOCAL)
        averages: list[float] = []
        for nid in node_ids:
            n = count_by[col].get(nid, 0)
            if n:
                averages.append(sum_by[col][nid] / n)
            else:
                averages.append(float("nan"))
        bundle[col] = averages
        n_ok = int(np.sum(np.isfinite(bundle[col].to_numpy(dtype=float))))
        progress(f"Stage {stage}: cached {col} ({n_ok:,}/{len(node_ids):,} nodes).")

    progress(
        f"Stage {stage}: bulk extract done — {used_elements:,} elements "
        f"(skipped={skipped})."
    )
    return bundle.reset_index(drop=True)


def _primary_layer_node_xyz(
    nodes,
    *,
    progress: Callable,
    stage: int,
) -> tuple[dict[Any, tuple[float, float, float]], int]:
    layer_counts: dict[int, int] = {}
    for node in nodes:
        try:
            layer = int(node.LayerIndex)
        except Exception:  # noqa: BLE001
            layer = 0
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    primary_layer = max(layer_counts, key=layer_counts.get) if layer_counts else 0
    if len(layer_counts) > 1:
        progress(
            f"Stage {stage}: multiple liner layers {sorted(layer_counts)}; "
            f"using layer {primary_layer}."
        )

    node_xyz: dict[Any, tuple[float, float, float]] = {}
    for node in nodes:
        try:
            layer = int(node.LayerIndex)
        except Exception:  # noqa: BLE001
            layer = primary_layer
        if layer != primary_layer:
            continue
        node_xyz[node.NodeID] = (
            float(node.XCoordinate),
            float(node.YCoordinate),
            float(node.ZCoordinate),
        )
    return node_xyz, primary_layer


def _extract_global_displacement_on_liner(
    model: "Model",
    *,
    stage: int,
    field_enum_name: str,
    field_label: str,
    srf_result_index: int,
    entity_name: str | None,
    rpc_timeout_s: float | None,
    progress: Callable,
) -> pd.DataFrame:
    """
    Global displacements on the liner: read solid-mesh nodal values (always
    global) and map onto liner node IDs.

    ``CompositeLinerResults.setCoordinateSystem('Global')`` fails on models that
    only define named frames (e.g. Tunnel Frames), which previously left results
    in the wrong axes and produced false discontinuities.
    """
    from rs3.results.ResultEnums import SolidsDataType

    solid_enum_name = _GLOBAL_DISPLACEMENT_ENUMS[field_enum_name]
    solid_type = SolidsDataType[solid_enum_name]

    results_api = model.Results
    available = _call_with_timeout(
        lambda: results_api.getResultsAvailability(
            stageNumber=stage,
            srfResultIndex=srf_result_index,
        ),
        rpc_timeout_s,
        f"getResultsAvailability(stage={stage})",
    )
    if not available:
        raise RuntimeError(f"No computed results for stage {stage}.")

    liner_list = _call_with_timeout(
        lambda: results_api.getCompositeLinerResults(
            stageNumber=[stage],
            srfResultIndex=srf_result_index,
        ),
        rpc_timeout_s,
        f"getCompositeLinerResults(stage={stage})",
    )
    if not liner_list:
        raise RuntimeError(f"getCompositeLinerResults returned no handles for stage {stage}.")
    liner = liner_list[0]

    progress(f"Stage {stage}: streaming liner node coordinates…")
    nodes = _call_with_timeout(
        lambda: liner.getLinerNodeResults(entityName=entity_name),
        rpc_timeout_s,
        f"getLinerNodeResults(stage={stage})",
    )
    if not nodes:
        raise RuntimeError(f"No liner nodes at stage {stage}.")

    node_xyz, _primary_layer = _primary_layer_node_xyz(
        nodes, progress=progress, stage=stage
    )
    progress(f"Stage {stage}: collected coordinates for {len(node_xyz):,} liner nodes.")

    progress(f"Stage {stage}: streaming global solid-mesh {field_label}…")
    mesh_list = _call_with_timeout(
        lambda: results_api.getMeshResults(
            stageNumber=[stage],
            srfResultIndex=srf_result_index,
            requiredDataTypes={solid_type},
        ),
        rpc_timeout_s,
        f"getMeshResults(stage={stage})",
    )
    if not mesh_list:
        raise RuntimeError(f"getMeshResults returned no handles for stage {stage}.")

    solid_nodes = _call_with_timeout(
        lambda: mesh_list[0].getMeshNodeResults(),
        rpc_timeout_s,
        f"getMeshNodeResults(stage={stage})",
    )
    if not solid_nodes:
        raise RuntimeError(f"No solid mesh nodes at stage {stage}.")

    value_by_id: dict[Any, float] = {}
    for snode in solid_nodes:
        try:
            value_by_id[snode.NodeID] = float(snode.getResult(solid_type))
        except Exception:  # noqa: BLE001
            continue

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    vals: list[float] = []
    missing = 0
    for nid, xyz in node_xyz.items():
        value = value_by_id.get(nid)
        if value is None or value != value:
            missing += 1
            continue
        xs.append(xyz[0])
        ys.append(xyz[1])
        zs.append(xyz[2])
        vals.append(value)

    if not vals:
        raise RuntimeError(
            f"Could not map any global {field_label} values onto liner nodes "
            f"at stage {stage} (missing={missing})."
        )

    progress(
        f"Stage {stage}: mapped global {field_label} onto {len(vals):,} liner nodes "
        f"(missing={missing})."
    )
    return pd.DataFrame({"X": xs, "Y": ys, "Z": zs, field_label: vals})


def _extract_stage(
    model: "Model",
    *,
    stage: int,
    data_type,
    field_label: str,
    coord: CoordSystem,
    srf_result_index: int,
    entity_name: str | None,
    unique_nodes: bool,
    rpc_timeout_s: float | None,
    progress: Callable,
) -> pd.DataFrame:
    results_api = model.Results
    available = _call_with_timeout(
        lambda: results_api.getResultsAvailability(
            stageNumber=stage,
            srfResultIndex=srf_result_index,
        ),
        rpc_timeout_s,
        f"getResultsAvailability(stage={stage})",
    )
    if not available:
        raise RuntimeError(f"No computed results for stage {stage}.")

    liner_list = _call_with_timeout(
        lambda: results_api.getCompositeLinerResults(
            stageNumber=[stage],
            srfResultIndex=srf_result_index,
        ),
        rpc_timeout_s,
        f"getCompositeLinerResults(stage={stage})",
    )
    if not liner_list:
        raise RuntimeError(f"getCompositeLinerResults returned no handles for stage {stage}.")
    liner = liner_list[0]

    _set_liner_coordinate_system(liner, coord, progress)

    progress(f"Stage {stage}: streaming liner node coordinates…")
    nodes = _call_with_timeout(
        lambda: liner.getLinerNodeResults(entityName=entity_name),
        rpc_timeout_s,
        f"getLinerNodeResults(stage={stage})",
    )
    if not nodes:
        raise RuntimeError(f"No liner nodes at stage {stage}.")

    node_xyz, primary_layer = _primary_layer_node_xyz(
        nodes, progress=progress, stage=stage
    )
    progress(f"Stage {stage}: collected coordinates for {len(node_xyz):,} liner nodes.")

    progress(f"Stage {stage}: streaming liner element results…")
    elements = _call_with_timeout(
        lambda: liner.getLinerElementResults(entityName=entity_name),
        rpc_timeout_s,
        f"getLinerElementResults(stage={stage})",
    )
    if not elements:
        raise RuntimeError(f"No liner elements with results at stage {stage}.")

    sum_by_node: dict[Any, float] = {}
    count_by_node: dict[Any, int] = {}
    skipped = 0
    used_elements = 0

    for el in elements:
        try:
            layer = int(el.LayerIndex)
        except Exception:  # noqa: BLE001
            layer = primary_layer
        if layer != primary_layer:
            skipped += 1
            continue

        attached = list(el.AttachedNodeIDs)
        raw_vals = _element_node_values(el, data_type)
        if not attached or len(raw_vals) != len(attached):
            skipped += 1
            continue

        used_elements += 1
        for nid, value in zip(attached, raw_vals):
            if nid not in node_xyz or value != value:
                skipped += 1
                continue
            sum_by_node[nid] = sum_by_node.get(nid, 0.0) + float(value)
            count_by_node[nid] = count_by_node.get(nid, 0) + 1

    if not sum_by_node:
        raise RuntimeError(
            f"Could not read any {field_label} values from liner elements "
            f"at stage {stage} (skipped={skipped})."
        )

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    vals: list[float] = []
    for nid, total in sum_by_node.items():
        xyz = node_xyz[nid]
        xs.append(xyz[0])
        ys.append(xyz[1])
        zs.append(xyz[2])
        vals.append(total / count_by_node[nid])

    progress(
        f"Stage {stage}: extracted {len(vals):,} unique liner nodes "
        f"from {used_elements:,} elements (skipped={skipped})."
    )

    if not unique_nodes:
        progress(
            f"Stage {stage}: note — returning unique nodal averages "
            "(per-element duplicates are not useful for contouring)."
        )

    return pd.DataFrame({"X": xs, "Y": ys, "Z": zs, field_label: vals})
