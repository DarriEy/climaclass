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


# Candidate variable names in remapped forcing stores (CF standard names first).
_TEMP_VARS = ("air_temperature", "airtemp", "temperature", "tas", "t2m", "tair", "Tair")
_PRECIP_VARS = ("precipitation_flux", "pptrate", "pr", "precipitation", "tp", "rainfall_flux")


def _pick_var(ds, candidates: Sequence[str], explicit: Optional[str], kind: str) -> str:
    if explicit:
        return explicit
    for c in candidates:
        if c in ds.data_vars:
            return c
    raise KeyError(f"No {kind} variable found (tried {list(candidates)}); pass it explicitly.")


def _to_celsius(da):
    """Convert a temperature DataArray to degrees Celsius using units, then magnitude."""
    units = str(da.attrs.get("units", "")).strip().lower()
    if units in ("k", "kelvin", "degk", "deg_k"):
        return da - 273.15
    if units in ("c", "degc", "celsius", "deg_c", "degrees_celsius"):
        return da
    # Ambiguous units: decide by magnitude of the first timestep (cheap).
    first_mean = float(da.isel(time=0).mean())
    return da - 273.15 if first_mean > 150.0 else da


def _precip_depth_per_step(da, dt_seconds: float):
    """Convert a precipitation DataArray to per-timestep depth in mm."""
    units = str(da.attrs.get("units", "")).strip().lower().replace(" ", "")
    if any(tok in units for tok in ("s-1", "s**-1", "/s")):  # a flux: kg m-2 s-1 == mm/s
        return da * dt_seconds
    if units in ("m", "meter", "metre"):  # depth in metres
        return da * 1000.0
    return da  # assume already a per-step depth in mm (kg m-2 == mm)


def _months_axis(da):
    """Return the non-month HRU dim name, asserting month runs 1..12 ascending."""
    return next(d for d in da.dims if d != "month")


def _forcing_climatology(forcing_files, temp_var, precip_var, id_coord, lat_coord):
    """Build per-HRU 12-month (T degC, P mm/month) climatology from a forcing store.

    Both fields reduce with a single ``groupby('time.month').mean()`` - a cheap
    streaming reduction. For precipitation given as a *rate* (the common case:
    ``kg m-2 s-1``) the monthly mean flux is multiplied by the seconds in each
    month to get a monthly total; only genuine per-step *accumulation* depths
    fall back to the heavier resample-and-sum.
    """
    import numpy as np
    import xarray as xr

    from .._types import DAYS_IN_MONTH

    files = sorted(forcing_files)
    ds = xr.open_mfdataset(files, combine="by_coords")
    tvar = _pick_var(ds, _TEMP_VARS, temp_var, "temperature")
    pvar = _pick_var(ds, _PRECIP_VARS, precip_var, "precipitation")

    temp_c = _to_celsius(ds[tvar]).groupby("time.month").mean("time")
    temp_arr = temp_c.transpose("month", _months_axis(temp_c)).values  # (12, n_hru), degC

    precip = ds[pvar]
    units = str(precip.attrs.get("units", "")).strip().lower().replace(" ", "")
    seconds_per_month = np.array(DAYS_IN_MONTH)[:, None] * 86400.0
    if any(tok in units for tok in ("s-1", "s**-1", "/s")):  # rate kg m-2 s-1 == mm/s
        flux = precip.groupby("time.month").mean("time")
        precip_arr = flux.transpose("month", _months_axis(flux)).values * seconds_per_month
    else:  # per-step accumulation depth -> sum within each calendar month
        times = ds["time"].values
        dt = float(np.median(np.diff(times)) / np.timedelta64(1, "s")) if len(times) > 1 else 3600.0
        depth = _precip_depth_per_step(precip, dt)
        monthly = depth.resample(time="1MS").sum().groupby("time.month").mean("time")
        precip_arr = monthly.transpose("month", _months_axis(monthly)).values

    # Static per-HRU coords from one file (avoids any concat-introduced time dim).
    static = xr.open_dataset(files[0])
    ids = np.asarray(static[id_coord].values).reshape(-1) if id_coord in static else None
    lats = np.asarray(static[lat_coord].values).reshape(-1) if lat_coord in static else None
    return ids, lats, temp_arr, precip_arr


def classify_forcing_store(
    forcing_files: Sequence,
    *,
    temp_var: Optional[str] = None,
    precip_var: Optional[str] = None,
    id_coord: str = "hruId",
    lat_coord: str = "latitude",
    elevation_by_id: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """Classify every HRU from a SYMFLUENCE *remapped forcing* store.

    This is the preferred path: it uses the meteorology already areal-weighted to
    each HRU (e.g. ``data/model_ready/forcings/*_remapped_*.nc``), so the
    classification is at true HRU resolution and consistent with what the model
    runs on - no WorldClim, no ~1 km resolution ceiling, no zonal statistics.

    Args:
        forcing_files: Remapped forcing NetCDF files (a multi-file time series
            with ``time`` x HRU variables and a per-HRU id/latitude coordinate).
        temp_var / precip_var: Override variable names if auto-detection (CF
            standard names, then common aliases) doesn't find them.
        id_coord: Per-HRU id coordinate (default ``"hruId"``).
        lat_coord: Per-HRU latitude coordinate (drives Thornthwaite PET).
        elevation_by_id: Optional ``{hru_id: mean_elevation_m}`` to enable the
            Holdridge altitudinal refinement (forcing stores carry no elevation).

    Returns:
        Flat ``{attribute: value}`` dict; lumped stores use unprefixed keys,
        distributed stores use ``HRU_{id}_`` prefixes.
    """
    import numpy as np

    ids, lats, temp_arr, precip_arr = _forcing_climatology(
        forcing_files, temp_var, precip_var, id_coord, lat_coord
    )
    n_hru = temp_arr.shape[1]
    lumped = n_hru == 1
    results: Dict[str, Any] = {}
    for i in range(n_hru):
        t12 = [float(x) for x in temp_arr[:, i]]
        p12 = [float(x) for x in precip_arr[:, i]]
        if any(np.isnan(t12)) or any(np.isnan(p12)):
            continue
        hru_id = int(ids[i]) if ids is not None else i
        latitude = float(lats[i]) if lats is not None else None
        elevation = elevation_by_id.get(hru_id) if elevation_by_id else None
        prefix = "" if lumped else f"HRU_{hru_id}_"
        results.update(
            record_to_attributes(t12, p12, latitude=latitude, elevation_m=elevation, prefix=prefix)
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
        # Prefer the remapped HRU-resolution forcing store; fall back to WorldClim.
        forcing = self._process_from_forcing_store()
        if forcing is not None:
            return forcing
        return self._process_from_worldclim()

    def _process_from_forcing_store(self) -> Optional[Dict[str, Any]]:
        """Classify from the remapped forcing store, if one exists (preferred)."""
        forcing_dir = self._get_data_path(  # type: ignore[attr-defined]
            "FORCING_MODEL_READY_PATH", "data/model_ready/forcings"
        )
        if not forcing_dir.exists():
            return None
        files = sorted(forcing_dir.glob("*_remapped_*.nc")) or sorted(forcing_dir.glob("*.nc"))
        if not files:
            return None

        elevation_by_id = self._elevation_by_id()
        self.logger.info(
            f"Classifying climate from {len(files)} remapped forcing file(s) at HRU resolution"
            f"{' with DEM altitudinal refinement' if elevation_by_id else ''}"
        )
        return classify_forcing_store(files, elevation_by_id=elevation_by_id)

    def _process_from_worldclim(self) -> Dict[str, Any]:
        """Fallback: zonal statistics over WorldClim monthly rasters."""
        import geopandas as gpd  # lazy, per SYMFLUENCE convention

        worldclim_path = self._get_data_path(  # type: ignore[attr-defined]
            "ATTRIBUTES_WORLDCLIM_PATH", "data/attributes/climate/worldclim"
        )
        if not worldclim_path.exists():
            self.logger.warning("No forcing store and no WorldClim data; skipping classification")
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
            f"Classifying climate from WorldClim for {len(catchment)} HRU(s)"
            f"{' with DEM altitudinal refinement' if dem_raster else ''}"
        )
        return classify_catchment(
            catchment, temp_files, precip_files, hru_id_field=hru_field, dem_raster=dem_raster
        )

    def _elevation_by_id(self) -> Optional[Dict[int, float]]:
        """Per-HRU mean elevation from the DEM, keyed by HRU id (for Holdridge)."""
        dem = self._find_dem()
        if dem is None:
            return None
        import geopandas as gpd

        catchment = gpd.read_file(self.catchment_path)  # type: ignore[attr-defined]
        hru_field = self._get_config_value(  # type: ignore[attr-defined]
            lambda: self.config.paths.catchment_hruid, default="HRU_ID", dict_key="CATCHMENT_SHP_HRUID"
        )
        ids = catchment[hru_field].tolist() if hru_field in catchment.columns else range(len(catchment))
        elevs = _zonal_means(catchment, dem)
        return {int(i): e for i, e in zip(ids, elevs) if e is not None}

    def _find_dem(self):
        """Locate the domain DEM for the Holdridge altitudinal refinement, if present."""
        dem_dir = self._get_data_path(  # type: ignore[attr-defined]
            "ATTRIBUTES_DEM_PATH", "data/attributes/elevation/dem"
        )
        dems = sorted(dem_dir.glob("*.tif")) if dem_dir.exists() else []
        return dems[0] if dems else None
