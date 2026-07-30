"""Liner result field definitions for tunnel visualization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CoordSystem(str, Enum):
    GLOBAL = "Global"
    LOCAL = "Local"


@dataclass(frozen=True)
class LinerField:
    """One extractable liner result column."""

    key: str
    label: str
    enum_name: str
    default_coord: CoordSystem
    csv_aliases: tuple[str, ...] = ()


# Keys match LinerResultTypes member names in rs3.results.ResultEnums.
LINER_FIELDS: tuple[LinerField, ...] = (
    LinerField(
        "displacement_z",
        "Z Displacement",
        "DISPLACEMENT_Z",
        CoordSystem.GLOBAL,
        ("Z Displacement", "Displacement Z", "DISPLACEMENT_Z"),
    ),
    LinerField(
        "displacement_x",
        "X Displacement",
        "DISPLACEMENT_X",
        CoordSystem.GLOBAL,
        ("X Displacement", "Displacement X", "DISPLACEMENT_X"),
    ),
    LinerField(
        "displacement_y",
        "Y Displacement",
        "DISPLACEMENT_Y",
        CoordSystem.GLOBAL,
        ("Y Displacement", "Displacement Y", "DISPLACEMENT_Y"),
    ),
    LinerField(
        "total_displacement",
        "Total Displacement",
        "TOTAL_DISPLACEMENT",
        CoordSystem.GLOBAL,
        ("Total Displacement", "TOTAL_DISPLACEMENT"),
    ),
    LinerField(
        "axial_force_x",
        "Axial Force X",
        "AXIAL_FORCE_XX",
        CoordSystem.LOCAL,
        ("Axial Force X", "AXIAL_FORCE_XX", "Axial Force XX"),
    ),
    LinerField(
        "axial_force_y",
        "Axial Force Y",
        "AXIAL_FORCE_YY",
        CoordSystem.LOCAL,
        ("Axial Force Y", "AXIAL_FORCE_YY", "Axial Force YY"),
    ),
    LinerField(
        "moment_y",
        "Moment Y",
        "MOMENT_YY",
        CoordSystem.LOCAL,
        ("Moment Y", "MOMENT_YY", "Moment YY"),
    ),
    LinerField(
        "moment_x",
        "Moment X",
        "MOMENT_XX",
        CoordSystem.LOCAL,
        ("Moment X", "MOMENT_XX", "Moment XX"),
    ),
    LinerField(
        "shear_force_xy",
        "Shear Force XY",
        "SHEAR_FORCE_XY",
        CoordSystem.LOCAL,
        ("Shear Force XY", "SHEAR_FORCE_XY"),
    ),
)

_BY_KEY = {f.key: f for f in LINER_FIELDS}


def get_field(key: str) -> LinerField:
    try:
        return _BY_KEY[key]
    except KeyError as exc:
        known = ", ".join(_BY_KEY)
        raise KeyError(f"Unknown liner field '{key}'. Choose one of: {known}") from exc


def dropdown_options() -> list[dict[str, str]]:
    return [{"label": f.label, "value": f.key} for f in LINER_FIELDS]


def cache_column_name(field_key: str, coord: str | CoordSystem) -> str:
    """Column name used in a multi-field extract bundle."""
    field = get_field(field_key)
    coord_val = coord.value if isinstance(coord, CoordSystem) else str(coord)
    return f"{field.label} [{coord_val}]"


def is_global_mesh_field(field_key: str) -> bool:
    """True if this field has a solid-mesh Global displacement source."""
    return get_field(field_key).enum_name in {
        "DISPLACEMENT_X",
        "DISPLACEMENT_Y",
        "DISPLACEMENT_Z",
        "TOTAL_DISPLACEMENT",
    }


def effective_coord_for_view(field_key: str, coord: str | CoordSystem) -> str:
    """
    Coord actually used for plotting.

    Global displacements come from the solid mesh. Forces/moments (and Local
    displacements) come from liner results in the model frame / Local axes.
    """
    coord_val = coord.value if isinstance(coord, CoordSystem) else str(coord)
    if coord_val == CoordSystem.GLOBAL.value and is_global_mesh_field(field_key):
        return CoordSystem.GLOBAL.value
    return CoordSystem.LOCAL.value


def dataframe_from_bundle(
    bundle: pd.DataFrame,
    field_key: str,
    coord: str | CoordSystem,
) -> pd.DataFrame:
    """Slice a 4-column view (X,Y,Z,value) from a cached multi-field bundle."""
    field = get_field(field_key)
    eff = effective_coord_for_view(field_key, coord)
    col = cache_column_name(field_key, eff)
    if col not in bundle.columns:
        # Fall back to the other axes if only one was cached.
        alt = cache_column_name(
            field_key,
            CoordSystem.LOCAL if eff == CoordSystem.GLOBAL.value else CoordSystem.GLOBAL,
        )
        if alt not in bundle.columns:
            raise KeyError(f"Cached results have no column for {field.label} [{eff}].")
        col = alt
    out = bundle[["X", "Y", "Z", col]].copy()
    out.columns = ["X", "Y", "Z", field.label]
    return out
