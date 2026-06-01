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

from .. import classify
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
    prefix: str = "",
) -> Dict[str, Any]:
    """Classify one HRU's climatology into flat ``climate.*`` attribute keys.

    Args:
        temp: 12 monthly mean temperatures [degC], Jan..Dec.
        precip: 12 monthly total precipitation [mm], Jan..Dec.
        latitude: Optional latitude to refine Thornthwaite PET.
        prefix: Optional key prefix for distributed domains. SYMFLUENCE expects
            ``"HRU_{id}_"`` so keys read ``HRU_3_climate.koppen_code``.

    Returns:
        Dict of attribute name -> value, ready to merge into a results dict.
    """
    climate = MonthlyClimate(temp=temp, precip=precip, latitude=latitude)
    results = classify(climate)
    k, h, t = results["koppen"], results["holdridge"], results["thornthwaite"]
    return {
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

        results: Dict[str, Any] = {}
        worldclim_path = self._get_data_path(  # type: ignore[attr-defined]
            "ATTRIBUTES_WORLDCLIM_PATH", "data/attributes/climate/worldclim"
        )
        if not worldclim_path.exists():
            self.logger.warning(f"WorldClim path not found, skipping classification: {worldclim_path}")
            return results

        temp = self._monthly_zonal(worldclim_path, "tavg")
        precip = self._monthly_zonal(worldclim_path, "prec")
        if temp is None or precip is None:
            self.logger.warning("Missing tavg/prec WorldClim rasters; skipping classification")
            return results

        # temp/precip are lists of 12-length climatologies, one per zone.
        for zone_idx, (t12, p12) in enumerate(zip(temp, precip)):
            if any(v is None for v in t12) or any(v is None for v in p12):
                continue
            prefix = "" if len(temp) == 1 else self._hru_prefix(zone_idx)
            results.update(record_to_attributes(t12, p12, prefix=prefix))
        return results

    # --- helpers -----------------------------------------------------------

    def _monthly_zonal(self, worldclim_path, var: str) -> Optional[List[List[Optional[float]]]]:
        """Return per-zone 12-month means for ``var`` via zonal statistics."""
        import geopandas as gpd  # lazy, per SYMFLUENCE convention
        from rasterstats import zonal_stats

        files = sorted(worldclim_path.glob(f"wc2.1_30s_{var}_*.tif"))
        if len(files) < 12:
            return None

        catchment = gpd.read_file(self.catchment_path)  # type: ignore[attr-defined]
        n_zones = len(catchment)
        monthly: List[List[Optional[float]]] = [[None] * 12 for _ in range(n_zones)]
        for month_idx, raster in enumerate(files[:12]):
            stats = zonal_stats(str(self.catchment_path), str(raster), stats=["mean"])  # type: ignore[attr-defined]
            for zone_idx, s in enumerate(stats):
                monthly[zone_idx][month_idx] = s.get("mean")
        return monthly

    def _hru_prefix(self, zone_idx: int) -> str:
        """Build the per-HRU attribute key prefix using the configured HRU id field."""
        import geopandas as gpd

        hru_field = self._get_config_value(  # type: ignore[attr-defined]
            lambda: self.config.paths.catchment_hruid, default="HRU_ID", dict_key="CATCHMENT_SHP_HRUID"
        )
        catchment = gpd.read_file(self.catchment_path)  # type: ignore[attr-defined]
        try:
            hru_id = catchment.iloc[zone_idx][hru_field]
        except Exception:  # noqa: BLE001
            hru_id = zone_idx
        return f"HRU_{hru_id}_"
