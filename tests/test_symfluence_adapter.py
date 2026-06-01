# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Darri Eythorsson

"""Tests for the SYMFLUENCE adapter's pure mapping helper.

The raster/zonal-stats path needs SYMFLUENCE + geospatial deps, so it is not
exercised here; the framework-independent mapping is.
"""

from climaclass.integrations.symfluence import record_to_attributes

ROME_T = [8.0, 8.8, 11.0, 13.8, 18.2, 22.3, 25.2, 25.1, 21.3, 16.7, 12.0, 8.9]
ROME_P = [80, 75, 70, 65, 50, 35, 20, 30, 70, 110, 110, 95]


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
