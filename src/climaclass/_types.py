# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Darri Eythorsson

"""Shared input/output data structures for :mod:`climaclass`.

The whole library is built around one small, explicit input record -
:class:`MonthlyClimate` - a 12-month climatology of temperature and
precipitation. Every classifier consumes that record and returns a typed
result, so the schemes stay interchangeable and trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

# Calendar constants. Index 0 == January, 11 == December throughout the package.
DAYS_IN_MONTH = (31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
# Day-of-year for the middle of each month (used for Thornthwaite day-length).
MID_MONTH_DOY = (16, 46, 75, 106, 136, 167, 197, 228, 259, 289, 320, 350)


def _validate_12(name: str, values: Sequence[float]) -> tuple:
    vals = tuple(float(v) for v in values)
    if len(vals) != 12:
        raise ValueError(f"{name} must have 12 monthly values (Jan..Dec), got {len(vals)}")
    return vals


@dataclass(frozen=True)
class MonthlyClimate:
    """A 12-month climatology for a single location or catchment.

    Args:
        temp: Monthly mean air temperature [degC], January..December.
        precip: Monthly total precipitation [mm], January..December.
        tmin: Optional monthly mean daily-minimum temperature [degC]. When
            absent it is approximated from ``temp`` where a scheme needs it.
        tmax: Optional monthly mean daily-maximum temperature [degC].
        latitude: Optional latitude [degrees, north positive]. Only used to
            refine the Thornthwaite day-length correction and to disambiguate
            the hemisphere; classification works without it.
    """

    temp: Sequence[float]
    precip: Sequence[float]
    tmin: Optional[Sequence[float]] = None
    tmax: Optional[Sequence[float]] = None
    latitude: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "temp", _validate_12("temp", self.temp))
        object.__setattr__(self, "precip", _validate_12("precip", self.precip))
        if self.tmin is not None:
            object.__setattr__(self, "tmin", _validate_12("tmin", self.tmin))
        if self.tmax is not None:
            object.__setattr__(self, "tmax", _validate_12("tmax", self.tmax))

    # --- convenient derived quantities (shared by several classifiers) ---

    @property
    def mat(self) -> float:
        """Mean annual temperature [degC]."""
        return sum(self.temp) / 12.0

    @property
    def map(self) -> float:
        """Mean annual precipitation [mm]."""
        return sum(self.precip)

    @property
    def hemisphere(self) -> str:
        """``"north"`` or ``"south"``, inferred from the seasonal temperature cycle.

        If ``latitude`` is given it wins; otherwise the warmer half-year decides.
        """
        if self.latitude is not None:
            return "north" if self.latitude >= 0 else "south"
        apr_sep = sum(self.temp[3:9]) / 6.0
        oct_mar = sum(self.temp[9:12] + self.temp[0:3]) / 6.0
        return "north" if apr_sep >= oct_mar else "south"


@dataclass(frozen=True)
class ClassificationResult:
    """A single scheme's verdict.

    Args:
        scheme: ``"koppen"`` | ``"holdridge"`` | ``"thornthwaite"``.
        code: Compact symbol (e.g. ``"Csb"``, ``"B"``).
        zone: Integer index into the scheme's canonical legend (or ``None``).
        name: Human-readable label (e.g. ``"Tropical rainforest"``).
        details: Scheme-specific intermediate values, for transparency/QA.
    """

    scheme: str
    code: str
    zone: Optional[int]
    name: str
    details: dict
