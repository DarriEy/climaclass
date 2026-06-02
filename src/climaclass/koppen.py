# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Darri Eythorsson

"""Koppen-Geiger climate classification from a 12-month climatology.

Implements the widely used formulation of Peel, Finlayson & McMahon (2007,
*HESS*) and the numeric legend (1..30) of Beck et al. (2018, *Sci. Data*),
which is the same legend rendered by the original Earth-Engine notebook this
package grew out of. Pure-Python: no NumPy, no Earth Engine.
"""

from __future__ import annotations

from ._types import ClassificationResult, MonthlyClimate

# Beck et al. (2018) numeric legend: code string -> integer zone.
KOPPEN_LEGEND = {
    "Af": 1, "Am": 2, "Aw": 3,
    "BWh": 4, "BWk": 5, "BSh": 6, "BSk": 7,
    "Csa": 8, "Csb": 9, "Csc": 10,
    "Cwa": 11, "Cwb": 12, "Cwc": 13,
    "Cfa": 14, "Cfb": 15, "Cfc": 16,
    "Dsa": 17, "Dsb": 18, "Dsc": 19, "Dsd": 20,
    "Dwa": 21, "Dwb": 22, "Dwc": 23, "Dwd": 24,
    "Dfa": 25, "Dfb": 26, "Dfc": 27, "Dfd": 28,
    "ET": 29, "EF": 30,
}

KOPPEN_NAMES = {
    "Af": "Tropical rainforest", "Am": "Tropical monsoon", "Aw": "Tropical savannah",
    "BWh": "Hot desert", "BWk": "Cold desert", "BSh": "Hot steppe", "BSk": "Cold steppe",
    "Csa": "Hot-summer Mediterranean", "Csb": "Warm-summer Mediterranean",
    "Csc": "Cold-summer Mediterranean",
    "Cwa": "Humid subtropical (dry winter)", "Cwb": "Subtropical highland (dry winter)",
    "Cwc": "Cold subtropical highland (dry winter)",
    "Cfa": "Humid subtropical", "Cfb": "Temperate oceanic", "Cfc": "Subpolar oceanic",
    "Dsa": "Hot-summer Mediterranean continental", "Dsb": "Warm-summer Mediterranean continental",
    "Dsc": "Mediterranean subarctic", "Dsd": "Mediterranean subarctic (severe winter)",
    "Dwa": "Hot-summer humid continental (dry winter)",
    "Dwb": "Warm-summer humid continental (dry winter)",
    "Dwc": "Subarctic (dry winter)", "Dwd": "Severe-winter subarctic (dry winter)",
    "Dfa": "Hot-summer humid continental", "Dfb": "Warm-summer humid continental",
    "Dfc": "Subarctic", "Dfd": "Severe-winter subarctic",
    "ET": "Tundra", "EF": "Ice cap",
}

# Official Köppen-Geiger colours (Beck et al. 2018), indexed by zone 1..30.
_KOPPEN_PALETTE = (
    "0000ff", "0078ff", "46aafa", "ff0000", "ff9696", "ffa929", "ffdc64",
    "ffff00", "c8c800", "969600", "96ff96", "64c864", "329632", "c8ff50",
    "64ff32", "32c800", "ff00ff", "c800c8", "963296", "966496", "aaafff",
    "5a78dc", "4b50b4", "320087", "00ffff", "32c8ff", "007d7d", "00465f",
    "b3b3b3", "666666",
)
#: Map of Köppen code (e.g. ``"Csb"``) -> hex colour, for consistent maps.
KOPPEN_COLORS = {code: f"#{_KOPPEN_PALETTE[zone - 1]}" for code, zone in KOPPEN_LEGEND.items()}

# Northern-hemisphere summer = Apr..Sep (indices 3..8); winter = Oct..Mar.
_NH_SUMMER = (3, 4, 5, 6, 7, 8)
_NH_WINTER = (9, 10, 11, 0, 1, 2)


def _seasonal_indices(hemisphere: str) -> tuple:
    """Return (summer_months, winter_months) as 0-based index tuples."""
    if hemisphere == "north":
        return _NH_SUMMER, _NH_WINTER
    return _NH_WINTER, _NH_SUMMER


def classify(climate: MonthlyClimate) -> ClassificationResult:
    """Classify a 12-month climatology into a Koppen-Geiger zone."""
    t = climate.temp
    p = climate.precip
    mat = climate.mat
    map_ = climate.map

    t_cold = min(t)
    t_hot = max(t)
    t_mon10 = sum(1 for x in t if x >= 10.0)
    p_dry = min(p)

    summer, winter = _seasonal_indices(climate.hemisphere)
    p_summer = sum(p[i] for i in summer)
    p_winter = sum(p[i] for i in winter)
    p_s_dry = min(p[i] for i in summer)
    p_w_dry = min(p[i] for i in winter)
    p_s_wet = max(p[i] for i in summer)
    p_w_wet = max(p[i] for i in winter)

    # Aridity threshold P_th (Peel et al. 2007).
    if p_winter >= 0.7 * map_:
        p_th = 2.0 * mat
    elif p_summer >= 0.7 * map_:
        p_th = 2.0 * mat + 28.0
    else:
        p_th = 2.0 * mat + 14.0

    details = {
        "MAT": round(mat, 2), "MAP": round(map_, 1),
        "Tcold": round(t_cold, 2), "Thot": round(t_hot, 2),
        "Tmon10": t_mon10, "Pdry": round(p_dry, 1),
        "Pthreshold": round(p_th, 2), "hemisphere": climate.hemisphere,
    }

    # --- Main class. Aridity (B) is tested first and overrides A/C/D; E is the
    #     cold residual. This ordering reproduces cold deserts (BWk) as mapped
    #     by Beck et al. (2018). ---
    if map_ < 10.0 * p_th:
        code = _arid(map_, p_th, mat)
    elif t_cold >= 18.0:
        code = _tropical(p_dry, map_)
    elif t_hot > 10.0 and t_cold > 0.0:
        code = "C" + _cd_second(p_s_dry, p_w_wet, p_w_dry, p_s_wet) + _c_third(t_hot, t_mon10)
    elif t_hot > 10.0 and t_cold <= 0.0:
        code = "D" + _cd_second(p_s_dry, p_w_wet, p_w_dry, p_s_wet) + _d_third(t_hot, t_mon10, t_cold)
    else:
        code = "ET" if t_hot >= 0.0 else "EF"

    return ClassificationResult(
        scheme="koppen",
        code=code,
        zone=KOPPEN_LEGEND.get(code),
        name=KOPPEN_NAMES.get(code, code),
        details=details,
    )


def _arid(map_: float, p_th: float, mat: float) -> str:
    second = "W" if map_ < 5.0 * p_th else "S"
    third = "h" if mat >= 18.0 else "k"
    return "B" + second + third


def _tropical(p_dry: float, map_: float) -> str:
    if p_dry >= 60.0:
        return "Af"
    if p_dry >= 100.0 - map_ / 25.0:
        return "Am"
    return "Aw"


def _cd_second(p_s_dry: float, p_w_wet: float, p_w_dry: float, p_s_wet: float) -> str:
    """Second letter for C/D climates: s (dry summer), w (dry winter) or f."""
    dry_summer = p_s_dry < 40.0 and p_s_dry < p_w_wet / 3.0
    dry_winter = p_w_dry < p_s_wet / 10.0
    if dry_summer and not dry_winter:
        return "s"
    if dry_winter and not dry_summer:
        return "w"
    # If both criteria trip, assign by the wetter season (Peel et al. 2007).
    if dry_summer and dry_winter:
        return "s" if p_w_wet >= p_s_wet else "w"
    return "f"


def _c_third(t_hot: float, t_mon10: int) -> str:
    if t_hot >= 22.0:
        return "a"
    if t_mon10 >= 4:
        return "b"
    return "c"


def _d_third(t_hot: float, t_mon10: int, t_cold: float) -> str:
    if t_hot >= 22.0:
        return "a"
    if t_mon10 >= 4:
        return "b"
    if t_cold <= -38.0:
        return "d"
    return "c"
