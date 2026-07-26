#!/usr/bin/env python3
"""
gold_valuation_fetch.py

Precomputes the full seven-framework gold valuation table server-side and emits
ONE compact JSON (data/gold/valuation.json) for Claude to read in a single fetch.

Why: doing this live costs ~9 web searches (~27k tokens) per query. Reading the
precomputed file costs ~800 tokens. Everything here changes monthly or slower --
only the spot gold price genuinely needs to be live, and that stays out of here
deliberately so the consumer fetches it fresh.

Requires env var FRED_API_KEY (same secret the sibling fred-macro-data repo uses).
"""
import os
import sys
import json
import datetime as dt
from pathlib import Path

import requests
import pandas as pd
import numpy as np

FRED_KEY = os.environ.get("FRED_API_KEY")
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
OUT = Path("data/gold/valuation.json")

OZ_PER_TONNE = 32150.7

# ---------------------------------------------------------------------------
# Slow-moving inputs. These are NOT available as clean long time series from a
# free API, so they are maintained here by hand with explicit as-of dates and a
# confidence flag. Refresh quarterly-ish. Confidence drives a per-row flag in
# the output so the consumer knows which numbers are authoritative vs. soft.
# ---------------------------------------------------------------------------
MANUAL = {
    "us_official_gold_tonnes":    {"v": 8133.5,  "asof": "2026-07", "src": "US Treasury",           "conf": "high"},
    "world_official_gold_tonnes": {"v": 37100.0, "asof": "2026-Q2", "src": "WGC/IMF IFS",           "conf": "medium"},
    "total_abovegroud_gold_tonnes": {"v": 221200.0, "asof": "2026-Q2", "src": "WGC (rolled fwd)",   "conf": "medium"},
    "eurozone_m2_eur_mn":         {"v": 16381796.0, "asof": "2026-05", "src": "ECB",                "conf": "high"},
    "china_m2_cny_bn":            {"v": 353670.0,   "asof": "2026-05", "src": "PBOC",               "conf": "high"},
    "global_sovereign_debt_usd_tn": {"v": 115.0,  "asof": "2025-Q3", "src": "IIF Global Debt Mon.", "conf": "medium"},
    "global_equity_mcap_usd_tn":  {"v": 154.0,    "asof": "2026-Q2", "src": "Statista/SIFMA",       "conf": "low"},
    "global_bond_outstanding_usd_tn": {"v": 147.0, "asof": "2026-Q1", "src": "SIFMA Fact Book",     "conf": "low"},
    "eurusd":                     {"v": 1.1386,   "asof": "2026-07", "src": "ECB ref",              "conf": "high"},
    "usdcny":                     {"v": 6.7592,   "asof": "2026-07", "src": "ECB ref",              "conf": "high"},
    # Gold price used ONLY to compute the historical ratio series and the
    # snapshot. Consumers should overwrite `spot_gold` with a live quote.
    "spot_gold_usd_oz":           {"v": 4070.8,   "asof": "2026-07", "src": "GC=F",                 "conf": "high"},
}

# Landmark gold prices for reconstructing pre-2000 ratio history. Without these
# the percentile window starts in 2000 and badly misrepresents where the true
# historical extremes sit (the 1980 monetary-crisis peak is the key one).
GOLD_LANDMARKS = [
    ("1971-08-01", 35.0), ("1974-12-01", 195.0), ("1976-08-01", 104.0),
    ("1980-01-01", 850.0), ("1982-06-01", 300.0), ("1985-01-01", 300.0),
    ("1987-12-01", 484.0), ("1990-01-01", 400.0), ("1996-02-01", 415.0),
    ("1999-08-01", 253.0), ("2001-04-01", 260.0),
]


def fred(series_id: str) -> pd.DataFrame:
    if not FRED_KEY:
        raise RuntimeError("FRED_API_KEY not set")
    r = requests.get(FRED_URL, params={
        "series_id": series_id, "api_key": FRED_KEY, "file_type": "json"
    }, timeout=30)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)


def gold_series() -> pd.DataFrame:
    """Landmark-interpolated gold price history back to 1971.

    Deliberately coarse: the goal is a defensible percentile/regime placement,
    not a tradeable price series. Monthly log-interpolation between landmarks
    understates interim volatility, so treat percentiles as approximate and
    never quote an interpolated point as a historical price.
    """
    pts = pd.DataFrame(GOLD_LANDMARKS, columns=["date", "close"])
    pts["date"] = pd.to_datetime(pts["date"])
    idx = pd.date_range("1971-08-01", dt.date.today(), freq="MS")
    s = pts.set_index("date")["close"].reindex(idx)
    s.iloc[-1] = MANUAL["spot_gold_usd_oz"]["v"]
    # log-space interpolation -- gold moves multiplicatively
    s = np.exp(np.log(s).interpolate(method="time"))
    return s.rename("close").reset_index().rename(columns={"index": "date"})


def pct_stats(series: pd.Series, current: float) -> dict:
    s = series.dropna()
    return {
        "current": round(float(current), 3),
        "percentile": round(float((s < current).mean() * 100), 1),
        "median": round(float(s.median()), 3),
        "mean": round(float(s.mean()), 3),
        "std": round(float(s.std()), 3),
        "min": round(float(s.min()), 3),
        "max": round(float(s.max()), 3),
        "zscore": round(float((current - s.mean()) / s.std()), 2),
        "n_obs": int(len(s)),
        "window": f"{s.index.min()} to {s.index.max()}" if hasattr(s.index, "min") else None,
    }


def main():
    print(f"[{dt.datetime.now():%Y-%m-%d %H:%M}] building gold valuation snapshot")

    m2 = fred("M2SL")            # $bn, monthly
    debt = fred("GFDEBTN")       # $mn, quarterly
    real = fred("DFII10")        # percent, daily

    g = MANUAL
    px = g["spot_gold_usd_oz"]["v"]
    g_us_oz = g["us_official_gold_tonnes"]["v"] * OZ_PER_TONNE
    g_off_oz = g["world_official_gold_tonnes"]["v"] * OZ_PER_TONNE
    g_tot_oz = g["total_abovegroud_gold_tonnes"]["v"] * OZ_PER_TONNE

    # ---- historical ratio series (frameworks 1 and 3) ---------------------
    gold = gold_series()
    h = pd.merge_asof(gold, m2.rename(columns={"value": "m2_bn"}), on="date", direction="backward")
    h = pd.merge_asof(h, debt.rename(columns={"value": "debt_mn"}), on="date", direction="backward")
    h["f1"] = (h.close * g_us_oz) / (h.m2_bn * 1e9) * 100
    h["f3"] = (h.close * g_us_oz) / (h.debt_mn * 1e6) * 100

    m2_now = float(m2.value.iloc[-1])
    debt_now = float(debt.value.iloc[-1])
    real_now = float(real.value.iloc[-1])

    f1_cur = px * g_us_oz / (m2_now * 1e9) * 100
    f3_cur = px * g_us_oz / (debt_now * 1e6) * 100

    # ---- current values for the non-historical frameworks -----------------
    ea_m2_usd = g["eurozone_m2_eur_mn"]["v"] * 1e6 * g["eurusd"]["v"]
    cn_m2_usd = g["china_m2_cny_bn"]["v"] * 1e9 / g["usdcny"]["v"]
    global_m2_usd = m2_now * 1e9 + ea_m2_usd + cn_m2_usd

    f2 = px * g_tot_oz / global_m2_usd * 100
    f5 = px * g_off_oz / (g["global_sovereign_debt_usd_tn"]["v"] * 1e12) * 100
    f6 = px * g_tot_oz / (g["global_equity_mcap_usd_tn"]["v"] * 1e12) * 100
    f7 = px * g_tot_oz / (g["global_bond_outstanding_usd_tn"]["v"] * 1e12) * 100

    def implied(target_ratio, denom_usd, oz):
        return round(target_ratio / 100 * denom_usd / oz, 0)

    s1 = pct_stats(h.f1, f1_cur)
    s3 = pct_stats(h.f3, f3_cur)
    s4 = pct_stats(real.value, real_now)

    out = {
        "generated_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "spot_gold_used": px,
        "spot_gold_note": "STALE BY DESIGN -- consumer must overwrite with a live quote and rescale.",
        "inputs": {k: {"value": v["v"], "asof": v["asof"], "source": v["src"], "confidence": v["conf"]}
                   for k, v in g.items()},
        "fred_asof": {
            "M2SL": str(m2.date.iloc[-1].date()),
            "GFDEBTN": str(debt.date.iloc[-1].date()),
            "DFII10": str(real.date.iloc[-1].date()),
        },
        "frameworks": {
            "1_us_m2_ratio": {
                **s1, "baseline_type": "computed",
                "history_from": "1971-08",
                "implied_gold_at_median": implied(s1["median"], m2_now * 1e9, g_us_oz),
                "direction": "lower_is_cheaper",
                "classification": "sovereign_monetary_credibility",
            },
            "2_global_m2_coverage": {
                "current": round(f2, 3), "baseline_type": "judgment_range",
                "baseline_note": "No long series for combined US+EA+CN M2 vs total gold. Range only.",
                "direction": "lower_is_cheaper",
                "classification": "broad_store_of_value",
                "scope_caveat": "World gold over THREE regions' M2 -- numerator global, denominator partial.",
                "definitional_caveat": "China M2 is closer to US M3 in construction; not strictly additive.",
            },
            "3_us_debt_coverage": {
                **s3, "baseline_type": "computed",
                "history_from": "1971-08",
                "implied_gold_at_median": implied(s3["median"], debt_now * 1e6, g_us_oz),
                "direction": "lower_is_cheaper",
                "classification": "sovereign_monetary_credibility",
            },
            "4_10y_real_yield": {
                **s4, "baseline_type": "computed",
                "history_from": "2003-01",
                "direction": "lower_is_MORE_EXPENSIVE (inverts vs all others)",
                "classification": "opportunity_cost_carry_signal",
                "note": "Not a coverage ratio. High percentile = high carry cost = headwind for gold.",
            },
            "5_global_sovereign_debt_coverage": {
                "current": round(f5, 3), "baseline_type": "judgment_range",
                "direction": "lower_is_cheaper",
                "classification": "sovereign_monetary_credibility",
                "overlap_caveat": "Sovereign debt is a SUBSET of the global bond market (framework 7). Not independent.",
            },
            "6_global_equities_ratio": {
                "current": round(f6, 3), "baseline_type": "judgment_range",
                "direction": "lower_is_cheaper",
                "classification": "broad_store_of_value",
                "confidence": "low -- source estimates for global equity cap vary by several $tn",
            },
            "7_global_bond_ratio": {
                "current": round(f7, 3), "baseline_type": "judgment_range",
                "direction": "lower_is_cheaper",
                "classification": "broad_store_of_value",
                "overlap_caveat": "Contains framework 5's sovereign debt. Not independent.",
            },
        },
        "structural_warnings": [
            "Above-ground gold grows ~1.5%/yr; M2 ~5%/yr. Every ratio carries a mechanical "
            "downward drift unrelated to valuation. Do not read long-horizon declines as cheapening.",
            "Frameworks 5 and 7 overlap: sovereign debt sits inside the global bond market.",
            "Vintage spread across inputs is up to ~4 months. Report per-input as-of dates.",
            "Pre-2000 gold history is landmark-interpolated -- percentiles are approximate.",
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    size = OUT.stat().st_size
    print(f"wrote {OUT} ({size} bytes, ~{size//4} tokens)")
    print(f"  F1 M2 ratio      {f1_cur:6.2f}%  p{s1['percentile']:.0f}  z{s1['zscore']:+.2f}")
    print(f"  F3 debt coverage {f3_cur:6.2f}%  p{s3['percentile']:.0f}  z{s3['zscore']:+.2f}")
    print(f"  F4 real yield    {real_now:6.2f}%  p{s4['percentile']:.0f}  z{s4['zscore']:+.2f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)
