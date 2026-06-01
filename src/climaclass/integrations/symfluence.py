# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Darri Eythorsson

"""SYMFLUENCE attribute-processor adapter for :mod:`climaclass`.

This module lets SYMFLUENCE emit Koppen-Geiger, Holdridge and Thornthwaite
classes as catchment attributes (``climate.koppen_*`` etc.), computed from the
same WorldClim monthly rasters SYMFLUENCE already acquires for its climate
attributes - no Earth Engine, no new data dependency.

It is intentionally decoupled:

* The pure mapping helper :func:`record_to_attributes` has no SYMFLUENCE
  dependency and can be unit-tested standalone.
* :class:`ClimateClassificationProcessor` only resolves the SYMFLUENCE base
  class at import time; if SYMFLUENCE is absent the class still imports (its
  base degrades to ``object``) so ``import climaclass`` never fails.

Install for use inside SYMFLUENCE with::

    pip install "climaclass[symfluence]"

and register it as an attribute processor (see the project README).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .. import holdridge, koppen, thornthwaite
from .._types import MonthlyClimate

# Resolve the SYMFLUENCE base class lazily/defensively so importing this module
# never hard-fails when SYMFLUENCE is not installed.
try:  # pragma: no cover - exercised only with SYMFLUENCE present
    from symfluence.data.preprocessing.attribute_processors.base import (
        BaseAttributeProcessor as _Base,
    )

    HAVE_SYMFLUENCE = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    _Base = object
    HAVE_SYMFLUENCE = False


def record_to_attributes(
    temp: Sequence[float],
    precip: Sequence[float],
    *,
    latitude: Optional[float] = None,
    elevation_m: Optional[float] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    """Classify one HRU's climatology into flat ``climate.*`` attribute keys.

    Args:
        temp: 12 monthly mean temperatures [degC], Jan..Dec.
        precip: 12 monthly total precipitation [mm], Jan..Dec.
        latitude: Optional HRU latitude; refines Thornthwaite PET day-length.
        elevation_m: Optional HRU mean elevation; enables the Holdridge
            altitudinal/latitudinal refinement.
        prefix: Optional key prefix for distributed domains. SYMFLUENCE expects
            ``"HRU_{id}_"`` so keys read ``HRU_3_climate.koppen_code``.

    Returns:
        Dict of attribute name -> value, ready to merge into a results dict.
    """
    climate = MonthlyClimate(temp=temp, precip=precip, latitude=latitude)
    k = koppen.classify(climate)
    h = holdridge.classify(climate, elevation_m=elevation_m)
    t = thornthwaite.classify(climate)

    attrs = {
        f"{prefix}climate.koppen_code": k.code,
        f"{prefix}climate.koppen_zone": k.zone,
        f"{prefix}climate.koppen_name": k.name,
        f"{prefix}climate.holdridge_zone": h.code,
        f"{prefix}climate.holdridge_code": h.zone,
        f"{prefix}climate.holdridge_belt": h.details.get("belt"),
        f"{prefix}climate.thornthwaite_code": t.code,
        f"{prefix}climate.thornthwaite_moisture_index": t.details.get("moisture_index"),
        f"{prefix}climate.thornthwaite_moisture_province": t.details.get("moisture_province"),
    }
    if elevation_m is not None:
        attrs[f"{prefix}climate.holdridge_altitudinal_belt"] = h.details.get("altitudinal_belt")
        attrs[f"{prefix}climate.holdridge_latitudinal_region"] = h.details.get("latitudinal_region")
        attrs[f"{prefix}climate.holdridge_is_altitudinal"] = h.details.get("is_altitudinal")
    return attrs


def _zonal_means(catchment, raster, stat: str = "mean") -> List[Optional[float]]:
    """Per-feature zonal statistic, reprojecting geometries to the raster CRS."""
    import rasterio
    from rasterstats import zonal_stats

    with rasterio.open(str(raster)) as src:
        rcrs = src.crs
    geoms = catchment.to_crs(rcrs) if (rcrs and catchment.crs and catchment.crs != rcrs) else catchment
    return [s.get(stat) for s in zonal_stats(list(geoms.geometry), str(raster), stats=[stat])]


def classify_catchment(
    catchment,
    temp_rasters: Sequence,
    precip_rasters: Sequence,
    *,
    hru_id_field: str = "HRU_ID",
    dem_raster: Optional[Any] = None,
) -> Dict[str, Any]:
    """Classify every HRU of a SYMFLUENCE hydrofabric catchment.

    This is the engine behind :class:`ClimateClassificationProcessor`, kept free
    of any SYMFLUENCE import so it can be tested directly. It needs only
    ``geopandas`` / ``rasterstats`` / ``rasterio`` (the ``[symfluence]`` extra).

    Args:
        catchment: A GeoDataFrame, or a path to a catchment shapefile. The
            SYMFLUENCE hydrofabric (``shapefiles/catchment/.../*_HRUs_*.shp``,
            EPSG:4326, one row per HRU) is the intended input.
        temp_rasters: 12 monthly mean-temperature rasters (any order; sorted by
            filename so ``*_01.tif`` .. ``*_12.tif`` map to Jan..Dec).
        precip_rasters: 12 monthly precipitation rasters.
        hru_id_field: Catchment field holding the integer HRU id (default
            ``"HRU_ID"``). Falls back to row order if the field is absent.
        dem_raster: Optional DEM; when given, per-HRU mean elevation drives the
            Holdridge altitudinal refinement.

    Returns:
        Flat ``{attribute: value}`` dict. Lumped (single-HRU) catchments use
        unprefixed keys; distributed catchments use ``HRU_{id}_`` prefixes,
        matching SYMFLUENCE's DataFrame builder.
    """
    import geopandas as gpd

    cat = catchment if isinstance(catchment, gpd.GeoDataFrame) else gpd.read_file(catchment)
    n = len(cat)
    if n == 0:
        return {}

    temp_files = sorted(temp_rasters)[:12]
    precip_files = sorted(precip_rasters)[:12]
    if len(temp_files) < 12 or len(precip_files) < 12:
        raise ValueError("classify_catchment needs 12 monthly temperature and 12 precipitation rasters")

    # Per-HRU monthly climatologies via zonal statistics.
    temp_monthly: List[List[Optional[float]]] = [[None] * 12 for _ in range(n)]
    precip_monthly: List[List[Optional[float]]] = [[None] * 12 for _ in range(n)]
    for m in range(12):
        for i, v in enumerate(_zonal_means(cat, temp_files[m])):
            temp_monthly[i][m] = v
        for i, v in enumerate(_zonal_means(cat, precip_files[m])):
            precip_monthly[i][m] = v

    # Per-HRU mean elevation (optional) for the Holdridge altitudinal refinement.
    elevation: List[Optional[float]] = [None] * n
    if dem_raster is not None:
        elevation = _zonal_means(cat, dem_raster)

    # Per-HRU latitude from the geographic-CRS centroid (for Thornthwaite PET).
    cat_geo = cat.to_crs(4326) if (cat.crs and cat.crs.to_epsg() != 4326) else cat
    latitudes = [geom.centroid.y for geom in cat_geo.geometry]

    ids = cat[hru_id_field].tolist() if hru_id_field in cat.columns else list(range(n))

    results: Dict[str, Any] = {}
    lumped = n == 1
    for i in range(n):
        t12, p12 = temp_monthly[i], precip_monthly[i]
        if any(v is None for v in t12) or any(v is None for v in p12):
            continue
        prefix = "" if lumped else f"HRU_{int(ids[i])}_"
        results.update(
            record_to_attributes(t12, p12, latitude=latitudes[i], elevation_m=elevation[i], prefix=prefix)
        )
    return results


class ClimateClassificationProcessor(_Base):
    """SYMFLUENCE attribute processor that adds climate-classification attributes.

    Reads 12 monthly WorldClim mean-temperature (``tavg``) and precipitation
    (``prec``) rasters, reduces them over the catchment (lumped) or each HRU
    (distributed) with zonal statistics, and classifies the resulting
    climatology with all three schemes.
    """

    #: SYMFLUENCE auto-discovery hooks (kept here so the framework side stays thin).
    name = "climate_classification"
    provides = ("climate.koppen_code", "climate.holdridge_zone", "climate.thornthwaite_code")

    def process(self) -> Dict[str, Any]:
        if not HAVE_SYMFLUENCE:  # pragma: no cover - guard for standalone import
            raise RuntimeError(
                "ClimateClassificationProcessor requires SYMFLUENCE. "
                "Install with: pip install 'climaclass[symfluence]'"
            )
        import geopandas as gpd  # lazy, per SYMFLUENCE convention

        worldclim_path = self._get_data_path(  # type: ignore[attr-defined]
            "ATTRIBUTES_WORLDCLIM_PATH", "data/attributes/climate/worldclim"
        )
        if not worldclim_path.exists():
            self.logger.warning(f"WorldClim path not found, skipping classification: {worldclim_path}")
            return {}

        temp_files = sorted(worldclim_path.glob("wc2.1_30s_tavg_*.tif"))
        precip_files = sorted(worldclim_path.glob("wc2.1_30s_prec_*.tif"))
        if len(temp_files) < 12 or len(precip_files) < 12:
            self.logger.warning("Missing tavg/prec WorldClim rasters; skipping classification")
            return {}

        catchment = gpd.read_file(self.catchment_path)  # type: ignore[attr-defined]  # read once
        hru_field = self._get_config_value(  # type: ignore[attr-defined]
            lambda: self.config.paths.catchment_hruid, default="HRU_ID", dict_key="CATCHMENT_SHP_HRUID"
        )
        dem_raster = self._find_dem()

        self.logger.info(
            f"Classifying climate for {len(catchment)} HRU(s)"
            f"{' with DEM altitudinal refinement' if dem_raster else ''}"
        )
        return classify_catchment(
            catchment, temp_files, precip_files, hru_id_field=hru_field, dem_raster=dem_raster
        )

    def _find_dem(self):
        """Locate the domain DEM for the Holdridge altitudinal refinement, if present."""
        dem_dir = self._get_data_path(  # type: ignore[attr-defined]
            "ATTRIBUTES_DEM_PATH", "data/attributes/elevation/dem"
        )
        dems = sorted(dem_dir.glob("*.tif")) if dem_dir.exists() else []
        return dems[0] if dems else None
