# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Darri Eythorsson

"""Thornthwaite climate classification from a 12-month climatology.

Implements the Thornthwaite (1948) potential-evapotranspiration model and the
moisture index. Potential evapotranspiration (PET) is computed from monthly
mean temperature via the classic heat-index formula, with the high-temperature
correction for T >= 26.5 degC and an optional latitude-based day-length
adjustment. Classification uses:

* **Moisture index** ``Im = 100 * (P - PET) / PET`` (the 1955 single-index form)
  -> humidity province (A perhumid .. E arid).
* **Thermal efficiency** = annual PET [mm] -> thermal province (A' .. E').

Pure Python; ``latitude`` only refines the day-length correction.
"""

from __future__ import annotations

import math

from ._types import DAYS_IN_MONTH, MID_MONTH_DOY, ClassificationResult, MonthlyClimate

# Moisture index thresholds (Im), wet..dry: (lower_bound, code, name).
_MOISTURE = (
    (100.0, "A", "Perhumid"),
    (80.0, "B4", "Humid (B4)"),
    (60.0, "B3", "Humid (B3)"),
    (40.0, "B2", "Humid (B2)"),
    (20.0, "B1", "Humid (B1)"),
    (0.0, "C2", "Moist subhumid"),
    (-33.3, "C1", "Dry subhumid"),
    (-66.7, "D", "Semiarid"),
    (float("-inf"), "E", "Arid"),
)

# Thermal-efficiency thresholds (annual PET, mm), warm..cold.
_THERMAL = (
    (1140.0, "A'", "Megathermal"),
    (997.0, "B'4", "Mesothermal (B'4)"),
    (855.0, "B'3", "Mesothermal (B'3)"),
    (712.0, "B'2", "Mesothermal (B'2)"),
    (570.0, "B'1", "Mesothermal (B'1)"),
    (427.0, "C'2", "Microthermal (C'2)"),
    (285.0, "C'1", "Microthermal (C'1)"),
    (142.0, "D'", "Tundra"),
    (float("-inf"), "E'", "Frost"),
)


def heat_index(temp: tuple) -> float:
    """Annual heat index I = sum((T/5)^1.514) over months with T > 0."""
    return sum((x / 5.0) ** 1.514 for x in temp if x > 0.0)


def _alpha(i: float) -> float:
    return 6.75e-7 * i**3 - 7.71e-5 * i**2 + 1.792e-2 * i + 0.49239


def _daylength_factor(latitude: float, month: int) -> float:
    """Correction factor (N/12)*(days/30) for the Thornthwaite PET adjustment."""
    phi = math.radians(latitude)
    decl = 0.409 * math.sin(2.0 * math.pi / 365.0 * MID_MONTH_DOY[month] - 1.39)
    x = -math.tan(phi) * math.tan(decl)
    x = max(-1.0, min(1.0, x))  # clamp for polar day/night
    omega = math.acos(x)
    daylight_hours = 24.0 / math.pi * omega
    return (daylight_hours / 12.0) * (DAYS_IN_MONTH[month] / 30.0)


def monthly_pet(climate: MonthlyClimate) -> list:
    """Monthly potential evapotranspiration [mm], Thornthwaite (1948)."""
    temp = climate.temp
    i = heat_index(temp)
    a = _alpha(i)
    pet = []
    for m, t in enumerate(temp):
        if t <= 0.0:
            unadjusted = 0.0
        elif t < 26.5:
            unadjusted = 16.0 * (10.0 * t / i) ** a if i > 0 else 0.0
        else:
            # High-temperature branch: PET saturates (Willmott et al. 1985).
            unadjusted = -415.85 + 32.24 * t - 0.43 * t * t
        if climate.latitude is not None:
            unadjusted *= _daylength_factor(climate.latitude, m)
        pet.append(max(unadjusted, 0.0))
    return pet


def _lookup(table: tuple, value: float) -> tuple:
    for lower, code, name in table:
        if value >= lower:
            return code, name
    return table[-1][1], table[-1][2]


def classify(climate: MonthlyClimate) -> ClassificationResult:
    """Classify a 12-month climatology by Thornthwaite moisture & thermal indices."""
    pet = monthly_pet(climate)
    annual_pet = sum(pet)
    precip = climate.map

    moisture_index = 100.0 * (precip - annual_pet) / annual_pet if annual_pet > 0 else float("inf")

    m_code, m_name = _lookup(_MOISTURE, moisture_index)
    t_code, t_name = _lookup(_THERMAL, annual_pet)

    code = f"{m_code}{t_code}"
    name = f"{m_name} / {t_name}"

    return ClassificationResult(
        scheme="thornthwaite",
        code=code,
        zone=None,
        name=name,
        details={
            "moisture_index": round(moisture_index, 1) if moisture_index != float("inf") else None,
            "annual_PET": round(annual_pet, 1),
            "annual_precip": round(precip, 1),
            "moisture_province": m_name,
            "thermal_province": t_name,
        },
    )
