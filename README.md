# climaclass

**Classical climate classification from monthly climatologies — pure Python, no Earth Engine required.**

`climaclass` turns a 12-month climatology of temperature and precipitation into a
climate class under three classical schemes:

| Scheme | Reference | Output |
|--------|-----------|--------|
| **Köppen–Geiger** | Peel et al. (2007); legend of Beck et al. (2018) | code (`Csb`) + zone 1–30 |
| **Holdridge life zones** | Holdridge (1967) | life-zone name + belt/province |
| **Thornthwaite** | Thornthwaite (1948) | moisture + thermal province |

It grew out of the [`Climate_Classifications`](https://github.com/DarriEy/Climate_Classifications)
Earth-Engine notebooks, but reimplements the *algorithms* as dependency-free
Python so they run anywhere — a laptop, a CI job, or inside a hydrological model
pipeline — without a Google Earth Engine account.

## Install

```bash
pip install climaclass                 # core: zero dependencies
pip install "climaclass[symfluence]"   # + SYMFLUENCE attribute-processor plugin
```

## Use

```python
from climaclass import MonthlyClimate, classify

reykjavik = MonthlyClimate(
    temp=[0.1, 0.4, 0.9, 3.3, 6.8, 9.4, 10.9, 10.5, 7.7, 4.5, 1.7, 0.4],   # °C, Jan..Dec
    precip=[76, 72, 82, 58, 44, 50, 52, 62, 67, 86, 73, 79],               # mm,  Jan..Dec
    latitude=64.1,
)

results = classify(reykjavik)
print(results["koppen"].code)        # 'Cfc'  (subpolar oceanic)
print(results["holdridge"].name)     # e.g. 'Boreal moist forest'
print(results["thornthwaite"].code)  # moisture/thermal province code
print(results["koppen"].details)     # MAT, MAP, Pthreshold, ... for QA
```

Run a single scheme directly:

```python
from climaclass import koppen
koppen.classify(reykjavik).zone      # 16  (Beck et al. legend)
```

### Input

Everything keys off one small record, `MonthlyClimate`:

- `temp` — 12 monthly **mean** temperatures (°C), January → December *(required)*
- `precip` — 12 monthly **total** precipitation (mm), January → December *(required)*
- `latitude` — optional; refines Thornthwaite PET day-length and fixes hemisphere
- `tmin` / `tmax` — optional, reserved for future scheme variants

## SYMFLUENCE plugin

The optional `symfluence` extra ships an attribute processor that emits climate
classes as catchment attributes (`climate.koppen_code`, `climate.holdridge_zone`,
`climate.thornthwaite_code`, …), computed from the **WorldClim** monthly rasters
SYMFLUENCE already acquires — so there is no new data dependency and no Earth
Engine in the loop. It is registered via the `symfluence.attribute_processors`
entry point and can be used as a regionalization / PUB grouping variable.

The integration is fully decoupled: importing `climaclass` never imports
SYMFLUENCE, and the pure mapping helper `record_to_attributes(temp, precip)` is
testable on its own.

## Develop

```bash
pip install -e ".[dev]"
pytest -q
ruff check src/
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). The
classification algorithms were prototyped in the author's
`Climate_Classifications` Earth-Engine notebooks and relicensed by the
copyright holder.
