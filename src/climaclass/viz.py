# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Darri Eythorsson

"""Map per-HRU climate classifications onto catchment geometry.

Takes the flat ``{attribute: value}`` dict produced by the SYMFLUENCE adapter
(or any per-HRU mapping) and renders categorical choropleth maps - one panel per
scheme - with proper legends and the official Köppen-Geiger colours.

Requires the ``[viz]`` extra (matplotlib) plus ``geopandas`` from ``[symfluence]``::

    pip install "climaclass[viz,symfluence]"
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence

from .koppen import KOPPEN_COLORS

# Which attribute column carries the categorical label for each scheme, plus a
# fixed colour map where one exists (Köppen) and a panel title.
_SCHEME = {
    "koppen": ("climate.koppen_code", KOPPEN_COLORS, "Köppen–Geiger"),
    "holdridge": ("climate.holdridge_zone", None, "Holdridge life zones"),
    "thornthwaite": ("climate.thornthwaite_moisture_province", None, "Thornthwaite moisture"),
}

_HRU_KEY = re.compile(r"^HRU_(\d+)_(climate\..+)$")


def attributes_to_frame(attributes: Dict[str, Any]):
    """Pivot flat ``climate.*`` attributes into a per-HRU DataFrame indexed by id.

    Handles both distributed keys (``HRU_3_climate.koppen_code``) and lumped keys
    (``climate.koppen_code`` -> a single row indexed 1).
    """
    import pandas as pd

    rows: Dict[int, Dict[str, Any]] = {}
    lumped: Dict[str, Any] = {}
    for key, value in attributes.items():
        m = _HRU_KEY.match(key)
        if m:
            rows.setdefault(int(m.group(1)), {})[m.group(2)] = value
        elif key.startswith("climate."):
            lumped[key] = value

    if rows:
        df = pd.DataFrame.from_dict(rows, orient="index")
    else:
        df = pd.DataFrame([lumped], index=[1])
    df.index.name = "hru_id"
    return df


def plot_classifications(
    catchment,
    attributes: Dict[str, Any],
    *,
    schemes: Sequence[str] = ("koppen", "holdridge", "thornthwaite"),
    hru_id_field: str = "HRU_ID",
    out_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
    dpi: int = 150,
    marker_size: float = 8.0,
):
    """Render one categorical map panel per scheme onto the catchment geometry.

    Auto-detects geometry: polygon catchments are filled choropleths; point
    catchments (HRU centroids) are rendered as a square-marker scatter - the
    robust choice when hydrofabric polygons are degenerate. See :func:`hru_points`.

    Args:
        catchment: A GeoDataFrame or path to the catchment shapefile.
        attributes: Flat per-HRU attribute dict (from the SYMFLUENCE adapter).
        schemes: Which schemes to draw (any of koppen/holdridge/thornthwaite).
        hru_id_field: Catchment field to join on (default ``"HRU_ID"``).
        out_path: If given, the figure is saved here.
        figsize: Figure size; defaults to ``(6 * n_schemes, 7)``.
        dpi: Save resolution.
        marker_size: Square-marker size for point catchments.

    Returns:
        The matplotlib ``Figure``.
    """
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    df = attributes_to_frame(attributes)
    cat = catchment if isinstance(catchment, gpd.GeoDataFrame) else gpd.read_file(catchment)
    merged = cat.merge(df, left_on=hru_id_field, right_index=True, how="left")
    is_point = bool(merged.geom_type.isin(["Point", "MultiPoint"]).all())

    n = len(schemes)
    fig, axes = plt.subplots(1, n, figsize=figsize or (6 * n, 7))
    axes = [axes] if n == 1 else list(axes)

    for ax, scheme in zip(axes, schemes):
        col, color_map, title = _SCHEME[scheme]
        cats = sorted(str(v) for v in merged[col].dropna().unique())
        if color_map:
            colors = [color_map.get(c, "#cccccc") for c in cats]
        else:
            cmap = plt.get_cmap("tab20", max(len(cats), 1))
            colors = [cmap(i) for i in range(len(cats))]
        cmap_by_cat = dict(zip(cats, colors))

        if is_point:
            # Robust to corrupt polygon rings: render HRU points/centroids.
            pts = merged.dropna(subset=[col])
            ax.scatter(
                pts.geometry.x, pts.geometry.y,
                c=[cmap_by_cat[str(v)] for v in pts[col]],
                s=marker_size, marker="s", linewidths=0,
            )
            ax.set_aspect("equal")
        else:
            plot_col = merged[col].astype("object")
            merged_cat = merged.assign(_cat=pd.Categorical(plot_col, categories=cats))
            merged_cat.plot(
                column="_cat", ax=ax, cmap=ListedColormap(colors), categorical=True,
                legend=False, linewidth=0, missing_kwds={"color": "lightgrey"},
            )
        handles = [Patch(facecolor=cmap_by_cat[c], edgecolor="none", label=c) for c in cats]
        ax.legend(
            handles=handles, fontsize=6, loc="center left",
            bbox_to_anchor=(1.0, 0.5), frameon=False, title=scheme,
        )
        ax.set_title(title)
        ax.set_axis_off()

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return fig


def hru_points(lon, lat, crs: str = "EPSG:4326", to_crs: Optional[str] = None):
    """Build a point GeoDataFrame of HRU centroids for spike-free mapping.

    Handy when the hydrofabric polygons are degenerate: pass the per-HRU
    longitude/latitude (e.g. from the remapped forcing store, or the catchment's
    ``center_lon``/``center_lat`` columns) and an ``HRU_ID`` column to join on.

    Args:
        lon, lat: Per-HRU coordinate sequences.
        crs: CRS of the input coordinates (default WGS84).
        to_crs: Optional CRS to reproject to (e.g. an equal-area projection).
    """
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=crs)
    return gdf.to_crs(to_crs) if to_crs else gdf
