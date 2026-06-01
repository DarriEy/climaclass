# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Darri Eythorsson

"""Behavioural tests for the three classification schemes.

Climatologies are approximate monthly normals for well-known stations; the
assertions pin the expected class so a regression in the threshold logic is
caught immediately.
"""

import pytest

from climaclass import MonthlyClimate, classify, holdridge, koppen, thornthwaite

# --- Reference climatologies (monthly normals, Jan..Dec) ---------------------

SINGAPORE = MonthlyClimate(  # equatorial, wet all year -> Af
    temp=[26.6, 27.2, 27.6, 28.0, 28.4, 28.3, 27.9, 27.8, 27.6, 27.5, 27.0, 26.6],
    precip=[243, 159, 186, 179, 171, 163, 159, 176, 170, 194, 257, 314],
    latitude=1.35,
)

ROME = MonthlyClimate(  # hot dry summer -> Csa
    temp=[8.0, 8.8, 11.0, 13.8, 18.2, 22.3, 25.2, 25.1, 21.3, 16.7, 12.0, 8.9],
    precip=[80, 75, 70, 65, 50, 35, 20, 30, 70, 110, 110, 95],
    latitude=41.9,
)

REYKJAVIK = MonthlyClimate(  # cool oceanic, no dry season -> Cfc
    temp=[0.1, 0.4, 0.9, 3.3, 6.8, 9.4, 10.9, 10.5, 7.7, 4.5, 1.7, 0.4],
    precip=[76, 72, 82, 58, 44, 50, 52, 62, 67, 86, 73, 79],
    latitude=64.1,
)

CAIRO = MonthlyClimate(  # hot desert -> BWh
    temp=[14, 15, 18, 22, 26, 28, 29, 28, 26, 24, 19, 15],
    precip=[5, 4, 3, 1, 1, 0, 0, 0, 0, 1, 3, 5],
    latitude=30.0,
)

VERKHOYANSK = MonthlyClimate(  # extreme continental Siberia -> Dfd (severe winter)
    temp=[-45, -42, -30, -13, 2, 13, 16, 12, 3, -15, -36, -44],
    precip=[7, 5, 5, 5, 9, 28, 33, 30, 17, 11, 10, 8],
    latitude=67.5,
)


# --- Koppen-Geiger -----------------------------------------------------------

@pytest.mark.parametrize(
    "clim, expected",
    [
        (SINGAPORE, "Af"),
        (ROME, "Csa"),
        (REYKJAVIK, "Cfc"),
        (CAIRO, "BWh"),
        (VERKHOYANSK, "Dfd"),
    ],
)
def test_koppen_codes(clim, expected):
    result = koppen.classify(clim)
    assert result.code == expected
    assert result.zone == koppen.KOPPEN_LEGEND[expected]
    assert result.scheme == "koppen"


def test_koppen_legend_is_complete_and_unique():
    zones = list(koppen.KOPPEN_LEGEND.values())
    assert sorted(zones) == list(range(1, 31))  # 30 distinct Beck et al. classes


# --- Holdridge ---------------------------------------------------------------

def test_holdridge_tropical_forest():
    # Singapore sits in the tropical belt; ~2570 mm/yr puts it in the moist/wet
    # forest provinces (true "rain forest" needs PET ratio < 0.25, i.e. >8000 mm).
    res = holdridge.classify(SINGAPORE)
    assert res.details["belt"] == "Tropical"
    assert "forest" in res.name.lower()


def test_holdridge_true_rainforest_needs_low_pet_ratio():
    wet = MonthlyClimate(temp=[27] * 12, precip=[800] * 12)  # ~9600 mm/yr
    res = holdridge.classify(wet)
    assert res.name == "Tropical rain forest"
    assert res.details["PET_ratio"] < 0.25


def test_holdridge_polar_desert():
    icecap = MonthlyClimate(temp=[-30] * 6 + [-5, -2, -8] + [-20] * 3, precip=[10] * 12)
    res = holdridge.classify(icecap)
    assert res.name == "Polar desert"
    assert res.details["biotemperature"] < 1.5


def test_holdridge_biotemperature_clamps():
    # Months below 0 and above 30 are clamped before averaging.
    clim = MonthlyClimate(temp=[-10, -10, -10, 40, 40, 40, 40, 40, 40, -10, -10, -10], precip=[100] * 12)
    # six months clamp to 30, six clamp to 0 -> biotemp == 15
    assert holdridge.biotemperature(clim) == pytest.approx(15.0)


# --- Thornthwaite ------------------------------------------------------------

def test_thornthwaite_arid_is_negative_index():
    res = thornthwaite.classify(CAIRO)
    assert res.details["moisture_index"] < -66.7  # arid province
    assert res.code.startswith("E")


def test_thornthwaite_humid_is_positive_index():
    res = thornthwaite.classify(REYKJAVIK)
    assert res.details["moisture_index"] > 0  # wet, low PET


def test_thornthwaite_pet_is_nonnegative():
    pet = thornthwaite.monthly_pet(VERKHOYANSK)
    assert len(pet) == 12
    assert all(v >= 0 for v in pet)  # frozen winter months -> 0, never negative


# --- Unified API & input validation ------------------------------------------

def test_classify_runs_all_schemes():
    out = classify(ROME)
    assert set(out) == {"koppen", "holdridge", "thornthwaite"}
    assert out["koppen"].code == "Csa"


def test_classify_subset():
    out = classify(ROME, schemes=["koppen"])
    assert set(out) == {"koppen"}


def test_unknown_scheme_raises():
    with pytest.raises(ValueError):
        classify(ROME, schemes=["bogus"])


def test_requires_12_months():
    with pytest.raises(ValueError):
        MonthlyClimate(temp=[10] * 11, precip=[50] * 12)


def test_hemisphere_inference():
    assert ROME.hemisphere == "north"
    southern = MonthlyClimate(temp=ROME.temp[6:] + ROME.temp[:6], precip=ROME.precip)
    assert southern.hemisphere == "south"
