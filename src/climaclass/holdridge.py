# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Darri Eythorsson

"""Holdridge life-zone classification from a 12-month climatology.

Follows Holdridge (1967), *Life Zone Ecology*. A life zone is the intersection
of three logarithmic axes:

* **Mean annual biotemperature** (degC) - mean of monthly temperatures each
  clamped to [0, 30]. Sets the latitudinal/altitudinal belt.
* **Mean annual precipitation** (mm).
* **Potential evapotranspiration ratio**, ``PET_ratio = 58.93 * biotemp / precip``.
  Sets the humidity province.

This mirrors the variables computed in the original Earth-Engine notebook
(``biotemperature``, ``pAnn``, ``PETR``) but evaluates them per-climatology in
pure Python rather than over a global image.
"""

from __future__ import annotations

from ._types import ClassificationResult, MonthlyClimate

# Biotemperature breakpoints (degC) -> belt name, ordered cold..hot.
# Each (upper_bound, belt) means: belt applies while biotemp < upper_bound.
_BELTS = (
    (1.5, "Polar"),
    (3.0, "Subpolar"),
    (6.0, "Boreal"),
    (12.0, "Cool temperate"),
    (18.0, "Warm temperate"),
    (24.0, "Subtropical"),
    (float("inf"), "Tropical"),
)

# Humidity provinces by PET ratio, ordered wet..dry. Index used to look up the
# zone name within a belt. (upper_bound, province) means province applies while
# PET_ratio < upper_bound.
_PROVINCES = (
    (0.25, "superhumid"),
    (0.50, "perhumid"),
    (1.00, "humid"),
    (2.00, "subhumid"),
    (4.00, "semiarid"),
    (8.00, "arid"),
    (16.0, "perarid"),
    (float("inf"), "superarid"),
)

# Per-belt life-zone names, indexed by humidity province (0 = wettest).
# Encodes the Holdridge hexagon; truncated rows repeat the driest realised zone.
_ZONES = {
    "Polar": [
        "Polar desert", "Polar desert", "Polar desert", "Polar desert",
        "Polar desert", "Polar desert", "Polar desert", "Polar desert",
    ],
    "Subpolar": [
        "Subpolar rain tundra", "Subpolar wet tundra", "Subpolar moist tundra",
        "Subpolar dry tundra", "Subpolar dry tundra", "Subpolar desert",
        "Subpolar desert", "Subpolar desert",
    ],
    "Boreal": [
        "Boreal rain forest", "Boreal wet forest", "Boreal moist forest",
        "Boreal dry scrub", "Boreal desert", "Boreal desert",
        "Boreal desert", "Boreal desert",
    ],
    "Cool temperate": [
        "Cool temperate rain forest", "Cool temperate wet forest",
        "Cool temperate moist forest", "Cool temperate steppe",
        "Cool temperate desert scrub", "Cool temperate desert",
        "Cool temperate desert", "Cool temperate desert",
    ],
    "Warm temperate": [
        "Warm temperate rain forest", "Warm temperate wet forest",
        "Warm temperate moist forest", "Warm temperate dry forest",
        "Warm temperate thorn steppe", "Warm temperate desert scrub",
        "Warm temperate desert", "Warm temperate desert",
    ],
    "Subtropical": [
        "Subtropical rain forest", "Subtropical wet forest",
        "Subtropical moist forest", "Subtropical dry forest",
        "Subtropical thorn woodland", "Subtropical desert scrub",
        "Subtropical desert", "Subtropical desert",
    ],
    "Tropical": [
        "Tropical rain forest", "Tropical wet forest", "Tropical moist forest",
        "Tropical dry forest", "Tropical very dry forest",
        "Tropical thorn woodland", "Tropical desert scrub", "Tropical desert",
    ],
}

# Stable integer codes: belt_index * 10 + province_index, plus 1 (1-based).
_BELT_ORDER = [b for _, b in _BELTS]

# Holdridge altitudinal belts, aligned index-for-index with _BELTS (cold..hot).
# The latitudinal *region* is read from sea-level biotemperature; the altitudinal
# *belt* from the actual (elevation-affected) biotemperature. When elevation
# cools a site below its latitudinal region, it ascends these belts.
_ALTITUDINAL_BELTS = (
    "Nival",        # Polar
    "Alpine",       # Subpolar
    "Subalpine",    # Boreal
    "Montane",      # Cool temperate
    "Lower montane",  # Warm temperate
    "Premontane",   # Subtropical
    "Basal",        # Tropical
)


def sea_level_biotemperature(climate: MonthlyClimate, elevation_m: float, lapse_rate_c_per_km: float) -> float:
    """Biotemperature reduced to sea level by removing the elevation lapse effect.

    Warms each monthly temperature back to its sea-level equivalent
    (``T + lapse * elevation_km``) before clamping to [0, 30] and averaging.
    Used to recover the *latitudinal region* independent of altitude.
    """
    warming = lapse_rate_c_per_km * (elevation_m / 1000.0)
    clamped = [min(max(x + warming, 0.0), 30.0) for x in climate.temp]
    return sum(clamped) / 12.0


def biotemperature(climate: MonthlyClimate) -> float:
    """Mean annual biotemperature [degC]: mean of monthly temps clamped to [0, 30]."""
    clamped = [min(max(x, 0.0), 30.0) for x in climate.temp]
    return sum(clamped) / 12.0


def _belt(biotemp: float) -> tuple:
    for idx, (upper, name) in enumerate(_BELTS):
        if biotemp < upper:
            return idx, name
    return len(_BELTS) - 1, _BELTS[-1][1]


def _province(pet_ratio: float) -> tuple:
    for idx, (upper, name) in enumerate(_PROVINCES):
        if pet_ratio < upper:
            return idx, name
    return len(_PROVINCES) - 1, _PROVINCES[-1][1]


def classify(
    climate: MonthlyClimate,
    elevation_m: float | None = None,
    lapse_rate_c_per_km: float = 6.0,
) -> ClassificationResult:
    """Classify a 12-month climatology into a Holdridge life zone.

    The life zone itself is always determined by the *actual* biotemperature,
    precipitation and PET ratio (elevation effects already live in the input
    temperatures). When ``elevation_m`` is supplied, the result additionally
    reports the Holdridge altitudinal/latitudinal distinction:

    * ``latitudinal_region`` - belt implied by the sea-level biotemperature.
    * ``altitudinal_belt`` - belt (Basal..Nival) implied by actual biotemperature.
    * ``is_altitudinal`` - True when altitude pushes the site into a colder belt
      than its latitudinal region (i.e. a montane/alpine/nival situation).

    Args:
        climate: The 12-month climatology.
        elevation_m: Optional mean elevation [m] for the altitudinal refinement.
        lapse_rate_c_per_km: Environmental lapse rate used to reduce temperature
            to sea level (default 6.0 degC/km, matching the source notebooks).
    """
    biotemp = biotemperature(climate)
    precip = climate.map
    # PET ratio is undefined for zero precipitation; treat as maximally arid.
    pet_ratio = (58.93 * biotemp / precip) if precip > 0 else float("inf")

    belt_idx, belt_name = _belt(biotemp)
    prov_idx, prov_name = _province(pet_ratio)
    name = _ZONES[belt_name][prov_idx]
    code = belt_idx * 10 + prov_idx + 1

    details = {
        "biotemperature": round(biotemp, 2),
        "annual_precip": round(precip, 1),
        "PET_ratio": round(pet_ratio, 3) if pet_ratio != float("inf") else None,
        "belt": belt_name,
        "humidity_province": prov_name,
    }

    if elevation_m is not None:
        sl_biotemp = sea_level_biotemperature(climate, elevation_m, lapse_rate_c_per_km)
        sl_idx, sl_region = _belt(sl_biotemp)
        details.update(
            elevation_m=round(float(elevation_m), 1),
            sea_level_biotemperature=round(sl_biotemp, 2),
            latitudinal_region=sl_region,
            altitudinal_belt=_ALTITUDINAL_BELTS[belt_idx],
            is_altitudinal=bool(belt_idx < sl_idx),
        )

    return ClassificationResult(
        scheme="holdridge",
        code=name,
        zone=code,
        name=name,
        details=details,
    )
