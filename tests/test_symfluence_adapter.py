# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Darri Eythorsson

"""Tests for the SYMFLUENCE adapter.

The pure mapping helper (`record_to_attributes`) is framework-independent and
always tested. The `classify_catchment` engine needs the geospatial stack
(geopandas / rasterstats / rasterio, the ``[symfluence]`` extra) and is tested
against a synthetic in-memory hydrofabric that mimics the SYMFLUENCE catchment
shapefile (EPSG:4326, one row per HRU, ``HRU_ID`` field).
"""

import pytest

from climaclass.integrations.symfluence import record_to_attributes

ROME_T = [8.0, 8.8, 11.0, 13.8, 18.2, 22.3, 25.2, 25.1, 21.3, 16.7, 12.0, 8.9]
ROME_P = [80, 75, 70, 65, 50, 35, 20, 30, 70, 110, 110, 95]

_HAVE_GEO = True
try:  # geospatial stack for the engine test
    import geopandas  # noqa: F401
    import numpy as np
    import rasterio  # noqa: F401
    from rasterio.transform import from_origin
    from shapely.geometry import box
except Exception:  # noqa: BLE001
    _HAVE_GEO = False

_HAVE_XR = True
try:
    import pandas as pd  # noqa: F401
    import xarray as xr  # noqa: F401
except Exception:  # noqa: BLE001
    _HAVE_XR = False

_HAVE_MPL = True
try:
    import matplotlib  # noqa: F401

    matplotlib.use("Agg")
except Exception:  # noqa: BLE001
    _HAVE_MPL = False


# --- pure mapping helper -----------------------------------------------------

def test_record_to_attributes_keys_and_koppen():
    attrs = record_to_attributes(ROME_T, ROME_P, latitude=41.9)
    assert attrs["climate.koppen_code"] == "Csa"
    assert attrs["climate.koppen_zone"] == 8
    assert "climate.holdridge_zone" in attrs
    assert "climate.thornthwaite_code" in attrs


def test_record_to_attributes_prefix():
    # SYMFLUENCE distributed-HRU key convention: HRU_{id}_climate.<attr>
    attrs = record_to_attributes(ROME_T, ROME_P, prefix="HRU_3_")
    assert all(k.startswith("HRU_3_climate.") for k in attrs)


def test_record_to_attributes_elevation_adds_altitudinal_keys():
    without = record_to_attributes(ROME_T, ROME_P)
    assert not any("altitudinal" in k for k in without)
    with_elev = record_to_attributes(ROME_T, ROME_P, elevation_m=2500.0)
    assert "climate.holdridge_altitudinal_belt" in with_elev
    assert "climate.holdridge_latitudinal_region" in with_elev
    assert "climate.holdridge_is_altitudinal" in with_elev


# --- classify_catchment engine (synthetic hydrofabric) ----------------------

pytestmark_geo = pytest.mark.skipif(not _HAVE_GEO, reason="geopandas/rasterio/rasterstats not installed")


def _write_constant_raster(path, value, bounds=(-20.0, 63.0, -18.0, 65.0), res=0.05):
    """Write a small EPSG:4326 GeoTIFF filled with a constant value."""
    minx, miny, maxx, maxy = bounds
    width = int(round((maxx - minx) / res))
    height = int(round((maxy - miny) / res))
    transform = from_origin(minx, maxy, res, res)
    data = np.full((height, width), float(value), dtype="float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data, 1)


@pytestmark_geo
def test_classify_catchment_distributed(tmp_path):
    """Two HRUs with different climate -> distinct per-HRU classes, HRU_ keys."""
    import geopandas as gpd

    from climaclass.integrations.symfluence import classify_catchment

    # Two side-by-side cells; HRU 1 cold/wet, HRU 2 warm/dry.
    hru1 = box(-20.0, 63.0, -19.0, 65.0)
    hru2 = box(-19.0, 63.0, -18.0, 65.0)
    cat = gpd.GeoDataFrame({"HRU_ID": [1, 2]}, geometry=[hru1, hru2], crs="EPSG:4326")

    # Build 12 monthly rasters per variable; left half vs right half differ.
    # Simplest: constant rasters but different per HRU via two-value raster.
    temp_files, precip_files = [], []
    for m in range(1, 13):
        # temperature: cold (2C) west, warm (24C) east
        tf = tmp_path / f"wc2.1_30s_tavg_{m:02d}.tif"
        _write_split_raster(tf, west=2.0, east=24.0)
        temp_files.append(tf)
        # precip: wet (180mm) west, dry (10mm) east
        pf = tmp_path / f"wc2.1_30s_prec_{m:02d}.tif"
        _write_split_raster(pf, west=180.0, east=10.0)
        precip_files.append(pf)

    out = classify_catchment(cat, temp_files, precip_files, hru_id_field="HRU_ID")

    assert out["HRU_1_climate.koppen_code"] != out["HRU_2_climate.koppen_code"]
    assert out["HRU_2_climate.koppen_code"].startswith("B")  # warm + dry east -> arid
    # latitude was derived per-HRU (both ~64N) -> Thornthwaite ran
    assert "HRU_1_climate.thornthwaite_code" in out


@pytestmark_geo
def test_classify_catchment_with_dem_adds_altitudinal(tmp_path):
    import geopandas as gpd

    from climaclass.integrations.symfluence import classify_catchment

    cat = gpd.GeoDataFrame(
        {"HRU_ID": [1]}, geometry=[box(-20.0, 63.0, -18.0, 65.0)], crs="EPSG:4326"
    )
    temp_files, precip_files = [], []
    for m in range(1, 13):
        tf = tmp_path / f"wc2.1_30s_tavg_{m:02d}.tif"
        _write_constant_raster(tf, 8.0)
        temp_files.append(tf)
        pf = tmp_path / f"wc2.1_30s_prec_{m:02d}.tif"
        _write_constant_raster(pf, 90.0)
        precip_files.append(pf)
    dem = tmp_path / "dem.tif"
    _write_constant_raster(dem, 2200.0)

    out = classify_catchment(cat, temp_files, precip_files, dem_raster=dem)
    # Single HRU -> unprefixed (lumped) keys.
    assert "climate.koppen_code" in out
    assert out["climate.holdridge_altitudinal_belt"] is not None
    assert "climate.holdridge_is_altitudinal" in out


def _write_split_raster(path, west, east, bounds=(-20.0, 63.0, -18.0, 65.0), res=0.05):
    """Raster whose western half == ``west`` and eastern half == ``east``."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    minx, miny, maxx, maxy = bounds
    width = int(round((maxx - minx) / res))
    height = int(round((maxy - miny) / res))
    data = np.empty((height, width), dtype="float32")
    data[:, : width // 2] = float(west)
    data[:, width // 2:] = float(east)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs="EPSG:4326", transform=from_origin(minx, maxy, res, res),
    ) as dst:
        dst.write(data, 1)


# --- classify_forcing_store (synthetic remapped forcing) --------------------

pytestmark_xr = pytest.mark.skipif(not _HAVE_XR, reason="xarray/netcdf4 not installed")


def _write_forcing(path, temps_c, precip_mm_per_day, lats, ids, n_days=730):
    """Write a synthetic remapped forcing file: time x hru, CF-named variables."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    time = pd.date_range("2000-01-01", periods=n_days, freq="D")
    nh = len(ids)
    air_t = np.empty((n_days, nh), dtype="float32")
    pflux = np.empty((n_days, nh), dtype="float32")
    for i in range(nh):
        air_t[:, i] = temps_c[i] + 273.15  # store in Kelvin
        pflux[:, i] = precip_mm_per_day[i] / 86400.0  # mm/day -> kg m-2 s-1
    ds = xr.Dataset(
        {
            "air_temperature": (("time", "hru"), air_t, {"units": "K"}),
            "precipitation_flux": (("time", "hru"), pflux, {"units": "kg m-2 s-1"}),
        },
        coords={
            "time": time,
            "hruId": ("hru", np.array(ids)),
            "latitude": ("hru", np.array(lats, dtype="float32")),
        },
    )
    ds.to_netcdf(path)


@pytestmark_xr
def test_classify_forcing_store_distributed(tmp_path):
    from climaclass.integrations.symfluence import classify_forcing_store

    f = tmp_path / "DOM_CARRA_remapped_2000-01-01.nc"
    # HRU 1: cold + wet (tundra/oceanic); HRU 2: warm + dry (arid).
    _write_forcing(f, temps_c=[3.0, 25.0], precip_mm_per_day=[5.0, 0.2], lats=[64.0, 64.0], ids=[1, 2])

    out = classify_forcing_store([f])
    assert out["HRU_1_climate.koppen_code"] != out["HRU_2_climate.koppen_code"]
    assert out["HRU_2_climate.koppen_code"].startswith("B")  # warm + dry -> arid
    assert "HRU_1_climate.thornthwaite_code" in out  # per-HRU latitude was used


@pytestmark_xr
def test_classify_forcing_store_elevation(tmp_path):
    from climaclass.integrations.symfluence import classify_forcing_store

    f = tmp_path / "DOM_remapped_2000.nc"
    _write_forcing(f, temps_c=[8.0], precip_mm_per_day=[3.0], lats=[46.0], ids=[7])
    out = classify_forcing_store([f], elevation_by_id={7: 2600.0})
    # Single HRU -> lumped (unprefixed) keys + altitudinal refinement present.
    assert "climate.koppen_code" in out
    assert out["climate.holdridge_altitudinal_belt"] is not None


# --- visualisation ----------------------------------------------------------

@pytest.mark.skipif(not (_HAVE_MPL and _HAVE_GEO), reason="matplotlib/geopandas not installed")
def test_plot_classifications_writes_png(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box

    from climaclass.viz import attributes_to_frame, plot_classifications

    cat = gpd.GeoDataFrame(
        {"HRU_ID": [1, 2]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )
    attrs = {
        "HRU_1_climate.koppen_code": "Cfc", "HRU_1_climate.holdridge_zone": "Boreal wet forest",
        "HRU_1_climate.thornthwaite_moisture_province": "Perhumid",
        "HRU_2_climate.koppen_code": "BWk", "HRU_2_climate.holdridge_zone": "Cool temperate desert",
        "HRU_2_climate.thornthwaite_moisture_province": "Arid",
    }
    frame = attributes_to_frame(attrs)
    assert list(frame.index) == [1, 2]
    assert frame.loc[1, "climate.koppen_code"] == "Cfc"

    out = tmp_path / "maps.png"
    fig = plot_classifications(cat, attrs, out_path=str(out))
    assert out.exists() and out.stat().st_size > 0
    import matplotlib.pyplot as plt

    plt.close(fig)


@pytest.mark.skipif(not (_HAVE_MPL and _HAVE_GEO), reason="matplotlib/geopandas not installed")
def test_plot_classifications_point_mode(tmp_path):
    from climaclass.viz import hru_points, plot_classifications

    pts = hru_points(lon=[-20.0, -18.0], lat=[64.0, 65.0])
    pts["HRU_ID"] = [1, 2]
    attrs = {
        "HRU_1_climate.koppen_code": "ET", "HRU_1_climate.holdridge_zone": "Subpolar rain tundra",
        "HRU_1_climate.thornthwaite_moisture_province": "Perhumid",
        "HRU_2_climate.koppen_code": "Cfc", "HRU_2_climate.holdridge_zone": "Boreal wet forest",
        "HRU_2_climate.thornthwaite_moisture_province": "Humid (B2)",
    }
    out = tmp_path / "points.png"
    fig = plot_classifications(pts, attrs, out_path=str(out))
    assert out.exists() and out.stat().st_size > 0
    import matplotlib.pyplot as plt

    plt.close(fig)
