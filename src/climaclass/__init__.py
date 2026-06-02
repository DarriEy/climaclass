# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Darri Eythorsson

"""climaclass - classical climate classification from monthly climatologies.

Three classical schemes, one tiny input record, zero heavy dependencies:

    >>> from climaclass import MonthlyClimate, classify
    >>> rome = MonthlyClimate(
    ...     temp=[8, 9, 11, 14, 18, 22, 25, 25, 21, 17, 12, 9],
    ...     precip=[80, 75, 70, 65, 50, 35, 20, 30, 70, 110, 110, 95],
    ... )
    >>> classify(rome)["koppen"].code
    'Csa'

Each classifier consumes a :class:`MonthlyClimate` and returns a
:class:`ClassificationResult`. The library is pure Python; the optional
SYMFLUENCE integration lives in :mod:`climaclass.integrations.symfluence` and is
only imported when you ask for it.
"""

from __future__ import annotations

from typing import Dict

from . import holdridge, koppen, thornthwaite
from ._types import ClassificationResult, MonthlyClimate

__version__ = "0.3.0"

__all__ = [
    "MonthlyClimate",
    "ClassificationResult",
    "classify",
    "koppen",
    "holdridge",
    "thornthwaite",
]

_SCHEMES = {
    "koppen": koppen.classify,
    "holdridge": holdridge.classify,
    "thornthwaite": thornthwaite.classify,
}


def classify(climate: MonthlyClimate, schemes=None) -> Dict[str, ClassificationResult]:
    """Run one or more classification schemes on a climatology.

    Args:
        climate: The 12-month climatology to classify.
        schemes: Iterable of scheme names to run. Defaults to all three
            (``"koppen"``, ``"holdridge"``, ``"thornthwaite"``).

    Returns:
        Mapping of scheme name -> :class:`ClassificationResult`.
    """
    if schemes is None:
        schemes = _SCHEMES.keys()
    out: Dict[str, ClassificationResult] = {}
    for name in schemes:
        try:
            out[name] = _SCHEMES[name](climate)
        except KeyError as exc:
            raise ValueError(f"Unknown scheme: {name!r}. Choose from {sorted(_SCHEMES)}") from exc
    return out
