# GTA VI Twitch CCU forecast — make.fun entry

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), as five quantiles per hour, from
everything recorded up to the end of hour 1. Competition:
`gta6-twitch-ccu-forecast-01` (make.fun, $250, 20 places).

## Entry

- `main.py` — the entrypoint (`python3 main.py`). Packed by the kit's
  official command (`node cli/makefun.mjs compress forecast.py --out
  main.py`); the readable source is `forecast.py` beside it.
- `forecast.py` — the model, written from scratch for this competition.
- `LICENSE` — MIT.

## Model (one paragraph)

Matched-comparable shape + scale regression, in log space. A rolling-
midnight console launch does not stay flat after hour 1: only Oceania,
Asia and Europe have the game at 00:00 UTC, the US unlocks at 05:00Z /
08:00Z, so the audience grows several-fold and peaks in the evening.
The program keeps only the comparables in the case file that match GTA
VI's launch (console-only, rolling midnight-local unlock, no early
access, hour-1 print >= 5,000), takes the median of their
ln(viewers_h / viewers_1) per hour, and adjusts it for this launch's own
scale with a shrunk, capped OLS slope of the log-ratios on log hour-1
print (a bigger hour-1 print leaves less headroom to grow). The spread
is the matched curves' own dispersion at that hour (std of the
log-ratios, floored 0.35, capped 1.0). Quantiles:
`p1 * exp(centre_h + z_tau * sigma_h)`, rounded half-up to 2 decimals.
If the strict filter leaves fewer than 4 curves it relaxes step by step,
and with no curve at all it falls back to the house persistence baseline,
so the program always writes a valid forecast. Standard library only, no
network, no randomness, deterministic.

## Validation (leave-one-out, matched population, practice case p00)

Mean pinball loss on the log scale, hours 2–24:

| forecaster | loss |
| --- | --- |
| house persistence baseline | 0.349 |
| this model, median shape only | 0.255 |
| this model, shape + scale regression | **0.169–0.172** |

Per-launch leave-one-out losses for the final configuration: 0.12–0.19
across all six matched launches. The practice cases' own answers are not
published and SullyGnome (the historical source) is behind a bot check,
so the leave-one-out result is the honest evidence for the model; see
the docstring in `forecast.py` for the limits (small matched set,
extrapolation beyond it for GTA VI's likely hour-1 print).

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Runs in ~0.25 s for the four practice cases, deterministic byte for byte.
