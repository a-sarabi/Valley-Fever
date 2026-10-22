"""Build weekly (MMWR) files for Maricopa, Pima and Pinal from the original data.

Steps, per county:

  1. AZMET daily weather files (``SSYYrd.txt``) -> daily table -> cleaning and filling -> MMWR-week means.
  2. EPA AirData daily PM10 files -> county/method filter -> mean per monitor, then over monitors per day
     -> MMWR-week means.
  3. ADHS "MMWR by County" workbooks -> weekly cases per county (newest workbook wins per year).
  4. Merge into ``Date, <COUNTY>, <25 weather columns, alphabetical>, Daily AQI Value,
     Daily Mean PM10 Concentration`` from 2006_1 to 2026_39 (``--end``).

Cleaning (details in README.md): physically impossible daily readings are set to missing (station rules,
neighbour-station rules for the 20" soil mean, documented sensor failures); soil-temperature gaps are filled from
the neighbour stations and every other gap by linear interpolation; PM10 is averaged per monitor-day before the
county mean, a monitor keeps reporting through its AirNow rows when its AQS rows stop, a day with too few
monitors is estimated from the reporting ones, and Pima uses the monitor types 79, 122 and 127; a trailing
partial case week is dropped.

Usage (from this folder):
    python preprocess_counties.py                        # all counties -> output/
    python preprocess_counties.py --counties Pima Pinal --end 2026_39
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import string
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent  # the repository's Data folder
OUT = HERE / "output"

# --------------------------------------------------------------------------------------------
# per-county settings
# --------------------------------------------------------------------------------------------
# pm10_methods: EPA AQS method codes of the monitors that are averaged (79 = R&P TEOM, the choice of
#   the original Maricopa script; 122 = Met One BAM; 127 = R&P Partisol 2025 filter samplers, used in
#   Tucson until 2017).
COUNTIES: dict[str, dict] = {
    "Maricopa": {
        "cases_col": "MARICOPA",
        "azmet_dir": DATA / "Maricopa" / "Weather Data MARICOPA" / "Original Data MARICOPA",
        "pm10_dir": DATA / "Maricopa" / "PM10 Data",
        "pm10_glob": "Maricopa *.csv",
        "pm10_county": "Maricopa",
        "pm10_methods": [79],
    },
    "Pima": {
        "cases_col": "PIMA",
        "azmet_dir": DATA / "Pima (Tucson)" / "Weather Data Pima" / "Original Data Pima",
        "pm10_dir": DATA / "Pima (Tucson)" / "PM10 Data",
        "pm10_glob": "Pima *.csv",
        "pm10_county": "Pima",
        "pm10_methods": [79, 122, 127],
    },
    "Pinal": {
        "cases_col": "PINAL",
        "azmet_dir": DATA / "Pinal (Coolidge)" / "Weather Data Coolidge" / "Original Data Coolidge",
        "pm10_dir": DATA / "Pinal (Coolidge)" / "PM10 Data",
        "pm10_county": "Pinal",
        "pm10_glob": "Pinal *.csv",
        "pm10_methods": [79],
    },
}

CASE_WORKBOOKS = [  # oldest first; a later workbook overrides the same MMWR year
    DATA / "Valley Fever Data" / "MMWR by County (2006-2024).xlsx",
    DATA / "Valley Fever Data" / "MMWR by County (2024-2025_ Updated).xlsx",
    DATA / "Valley Fever Data" / "MMWR by County (2025-26_ MMWRWK-YR).xlsx",
]



# last MMWR week of every county file: the last complete week of case counts (2026_40 is partial)
DEFAULT_END = "2026_39"

FINAL_WEATHER_COLS = [  # alphabetical, as in the original Maricopa file
    '20" Soil Temp - Max', '20" Soil Temp - Mean', '20" Soil Temp - Min',
    '4" Soil Temp - Max', '4" Soil Temp - Mean', '4" Soil Temp - Min',
    'Actual Vapor Pressure - Daily Mean', 'Air Temp - Max', 'Air Temp - Mean', 'Air Temp - Min',
    'Dewpoint - Daily Mean', 'Heat Units', 'Max Wind Speed', 'Precipitation - Total',
    'RH - Max', 'RH - Mean', 'RH - Min', 'Reference ET', 'Reference ET Penman-Monteith',
    'Solar Rad. - Total', 'VPD - Mean', 'Wind Direction Std Dev', 'Wind Speed - Mean',
    'Wind Vector Direction', 'Wind Vector Magnitude',
]
PM10_COLS = ["Daily AQI Value", "Daily Mean PM10 Concentration"]
SOIL20 = ['20" Soil Temp - Max', '20" Soil Temp - Min', '20" Soil Temp - Mean']
WIND_DIR = ["Wind Direction Std Dev", "Wind Vector Direction", "Wind Vector Magnitude"]

# Daily weather plausibility rules (recommended mode). A flagged value is set to missing and then
# filled (20" soil from the neighbour stations, everything else by linear interpolation).
# Thresholds were calibrated on the three stations (see README.md).
SOIL4 = ['4" Soil Temp - Max', '4" Soil Temp - Min', '4" Soil Temp - Mean']
TRIPLES = [  # (max, mean, min) columns whose order is enforced on filled days
    ("Air Temp - Max", "Air Temp - Mean", "Air Temp - Min"),
    ("RH - Max", "RH - Mean", "RH - Min"),
    ('4" Soil Temp - Max', '4" Soil Temp - Mean', '4" Soil Temp - Min'),
    ('20" Soil Temp - Max', '20" Soil Temp - Mean', '20" Soil Temp - Min'),
]
QC_RULES = {
    # station-level rules
    "rain_exempt_mm": 10.0,             # the two 20" range rules are not applied on a cooling day with at least this much rain on it or the day before
    "soil20_daily_range_gt": 4.0,       # 20" soil Max - Min above this (deg C) -> the three 20" values missing
    "soil20_within_day_gt": 4.0,        # 20" Max - Mean or Mean - Min above this -> missing (catches days whose Min or Max is 999)
    "soil20_below": 0.0,                # any 20" soil value below this -> missing
    "soil4max_minus_airmax_gt": 20.0,   # 4" soil Max more than this above the air Max -> the three 4" values missing
    "soil4mean_minus_airmean_gt": 12.0, # 4" soil Mean more than this above the air Mean -> the three 4" values missing
    "max_wind_speed_gt": 35.0,          # daily Max Wind Speed (m/s) above this -> missing
    "wind_dir_sd_below": 15.0,          # stuck vane: wind-direction std dev below this AND ...
    "stuck_vane_ratio": 0.999,          #   ... vector magnitude >= this x mean speed (or adjoining such a day) -> direction columns missing
    "rh_clip": (0.0, 100.0),            # relative humidity clipped to this range
    # neighbour rules for the 20" mean (each station is compared with every other station; a day is flagged only
    # when it disagrees with all of them, so that a fault at one neighbour cannot flag a clean station)
    "soil20_jump_gt": 3.0,              # change since the last valid day (at most 5 days back) minus the neighbour's change over the same days
    "soil20_decay_gt": 1.5,             # after a flagged run, days stay missing while they differ by more than this from the last
    "soil20_decay_max_days": 21,        #   clean pre-run value moved by the neighbour's change; at most this many days,
    "soil20_decay_after_run_le": 10,    #   and only after runs of at most this many days (a spike, not an outage)
}
# Documented sensor failures that the generic rules do not catch completely: (county, columns, first day, last day, reason).
MANUAL_EXCLUSIONS = [
    ("Pinal", SOIL20, "2021-08-24", "2021-11-13",
     "Coolidge 20\" soil sensor failed on 2021-08-24 (daily range 10-20 C, then 5-6 C below the pre-failure level "
     "while the 4\" sensor was unchanged) and read 999 from 2021-09-12 until it was replaced on 2021-11-14"),
    ("Pima", SOIL4, "2006-07-27", "2006-07-31",
     "Tucson 4\" soil sensor read maxima of 48-60 C and means 8-12 C above the air mean on five rainy days"),
    ("Pinal", SOIL4 + SOIL20, "2019-02-17", "2019-02-21",
     "Coolidge soil sensors spiked together on 2019-02-18..20 (4\" mean up to 20 C above the air mean, 20\" to 32 C); "
     "the onset day 02-17 and the tail day 02-21 are excluded with them so that they do not anchor the fill"),
]
SOIL20_FILL_MAX_DAYS = 120   # soil gaps up to this length are filled from the neighbour stations (longer ones linearly)
SPREAD_WINDOW_DAYS = 7       # 20" Max/Min of a filled day use the median spread of up to this many clean days on each side
NEIGHBOUR_REF_INTERP_DAYS = 7  # a neighbour's own gaps up to this length are linearly interpolated before it serves as a reference
PM10_MIN_MONITOR_FRAC = 0.3   # a day is used only if it has at least max(2, ceil(frac * yearly median)) monitors
CASES_PARTIAL_WEEK_FRAC = 0.5  # a trailing week with a statewide total below this fraction of the median of the previous 8 weeks is dropped


def log(msg: str) -> None:
    print(msg, flush=True)


def save_text(text: str, path: Path) -> Path:
    """Write ``text`` to ``path``; if the file is locked, write ``<name>.new<suffix>`` instead."""
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        alt = path.with_name(path.stem + ".new" + path.suffix)
        alt.write_text(text, encoding="utf-8")
        log(f"  WARNING: {path.name} is locked by another program; written to {alt.name} instead")
        return alt


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write ``df`` to ``path``; if the file is locked (open in Excel), write ``<name>.new.csv`` instead."""
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        alt = path.with_name(path.stem + ".new.csv")
        df.to_csv(alt, index=False)
        log(f"  WARNING: {path.name} is locked by another program; written to {alt.name} instead")
        return alt


# --------------------------------------------------------------------------------------------
# MMWR weeks
# --------------------------------------------------------------------------------------------
def mmwr_year_week(date: datetime) -> tuple[int, int]:
    """MMWR year and week of ``date`` (CDC definition; copied from the original scripts).

    Weeks run Sunday-Saturday; week 1 is the first week of the year with at least four days in
    January, i.e. the week that contains January 4.
    """
    week_start = date - timedelta(days=(date.weekday() + 1) % 7)
    mmwr_year = date.year
    january_4 = datetime(mmwr_year, 1, 4)
    anchor_sunday = january_4 - timedelta(days=(january_4.weekday() + 1) % 7)
    if week_start < anchor_sunday:
        mmwr_year -= 1
        january_4_prev = datetime(mmwr_year, 1, 4)
        anchor_sunday = january_4_prev - timedelta(days=(january_4_prev.weekday() + 1) % 7)
    delta_days = (week_start - anchor_sunday).days
    mmwr_week = delta_days // 7 + 1
    dec_31 = datetime(mmwr_year, 12, 31)
    last_sunday = dec_31 - timedelta(days=(dec_31.weekday() + 1) % 7)
    days_in_last_week = (dec_31 - last_sunday).days + 1
    if days_in_last_week < 4 and week_start >= last_sunday:
        mmwr_week = 1
        mmwr_year += 1
    return mmwr_year, mmwr_week


def mmwr_year_week_reference(date: datetime) -> tuple[int, int]:
    """Independent implementation used only to self-test ``mmwr_year_week``.

    A Sunday-Saturday week belongs to the MMWR year in which at least four of its days fall,
    i.e. the calendar year of its Wednesday; the week number counts Sundays since the week that
    contains January 4 of that year (the first week with four or more January days).
    """
    sunday = date - timedelta(days=(date.weekday() + 1) % 7)
    wednesday = sunday + timedelta(days=3)
    year = wednesday.year
    jan4 = datetime(year, 1, 4)
    first_sunday = jan4 - timedelta(days=(jan4.weekday() + 1) % 7)
    return year, (sunday - first_sunday).days // 7 + 1


def self_test_mmwr(start: int = 1985, end: int = 2030) -> None:
    d = datetime(start, 1, 1)
    bad = []
    while d.year <= end:
        if mmwr_year_week(d) != mmwr_year_week_reference(d):
            bad.append(d)
        d += timedelta(days=1)
    if bad:
        raise SystemExit(f"MMWR self-test failed for {len(bad)} dates, e.g. {bad[:3]}")
    known = {datetime(2025, 12, 28): (2025, 53), datetime(2026, 1, 4): (2026, 1),
             datetime(2020, 12, 31): (2020, 53), datetime(2021, 1, 3): (2021, 1),
             datetime(2006, 1, 1): (2006, 1), datetime(2014, 12, 28): (2014, 53)}
    for k, v in known.items():
        assert mmwr_year_week(k) == v, (k, mmwr_year_week(k), v)


def week_label(year: int, week: int) -> str:
    return f"{int(year)}_{int(week)}"


def weeks_in_mmwr_year(year: int) -> int:
    return mmwr_year_week(datetime(year, 12, 28))[1]  # December 28 is always in the last week


def mmwr_calendar(first: tuple[int, int], last: tuple[int, int]) -> pd.DataFrame:
    """All MMWR (year, week) pairs from ``first`` to ``last`` inclusive, in order."""
    rows = []
    year, week = first
    while (year, week) <= last:
        rows.append((year, week))
        if week < weeks_in_mmwr_year(year):
            week += 1
        else:
            year, week = year + 1, 1
    return pd.DataFrame(rows, columns=["MMWR_Year", "MMWR_Week"])


def add_mmwr(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    yw = df[date_col].apply(mmwr_year_week)
    df["MMWR_Year"] = [y for y, _ in yw]
    df["MMWR_Week"] = [w for _, w in yw]
    return df


# --------------------------------------------------------------------------------------------
# 1. AZMET daily weather -> MMWR weekly means
# --------------------------------------------------------------------------------------------
UNIFIED_HEADERS = [
    "Year", "Day of Year", "Station Number", "Air Temp - Max", "Air Temp - Min", "Air Temp - Mean",
    "RH - Max", "RH - Min", "RH - Mean", "VPD - Mean", "Solar Rad. - Total", "Precipitation - Total",
    '4" Soil Temp - Max', '4" Soil Temp - Min', '4" Soil Temp - Mean',
    '20" Soil Temp - Max', '20" Soil Temp - Min', '20" Soil Temp - Mean',
    "Wind Speed - Mean", "Wind Vector Magnitude", "Wind Vector Direction",
    "Wind Direction Std Dev", "Max Wind Speed", "Heat Units", "Reference ET",
    "Reference ET (Penman-Monteith)", "Actual Vapor Pressure - Daily Mean", "Dewpoint - Daily Mean",
]
N_FIELDS_2003 = 28   # 2003-present AZMET daily format
N_FIELDS_1987 = 25   # 1987-2002 format (2" and 4" soil temperature, no vapour pressure / dew point)
PRINTABLE = set(string.printable)
META_COLS = ("Year", "Day of Year", "Station Number")


def azmet_file_year(name: str) -> int | None:
    """``0624rd.txt`` -> 2024, ``0687rd.txt`` -> 1987 (station = first two characters)."""
    m = re.fullmatch(r"(\d{2})(\d{2})rd\.txt", name, flags=re.IGNORECASE)
    if not m:
        return None
    yy = int(m.group(2))
    return 1900 + yy if yy >= 87 else 2000 + yy


def parse_azmet_folder(folder: Path) -> tuple[pd.DataFrame, dict]:
    """Parse every ``SSYYrd.txt`` file in ``folder`` into one daily table (original parsing logic)."""
    rows, info = [], {"files": 0, "skipped_lines": 0, "duplicate_lines": 0, "skipped_examples": []}
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".txt"):
            continue
        year = azmet_file_year(name)
        if year is None:
            log(f"  skipping {name}: cannot read the year from the file name")
            continue
        n_fields = N_FIELDS_1987 if year <= 2002 else N_FIELDS_2003
        info["files"] += 1
        prev = None
        with open(folder / name, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                cleaned = "".join(ch for ch in line if ch in PRINTABLE)
                data = cleaned.strip().split(",")
                if len(data) != n_fields:
                    if cleaned.strip():  # a blank last line is expected and not reported
                        info["skipped_lines"] += 1
                        if len(info["skipped_examples"]) < 5:
                            info["skipped_examples"].append(f"{name}: {cleaned.strip()[:60]}")
                    continue
                if year <= 2002:
                    if len(data[0]) == 2:
                        data[0] = "19" + data[0]
                    data += [None, None, None]
                    data = data[:24] + [data[25], data[24]] + data[26:]
                if data == prev:
                    info["duplicate_lines"] += 1
                    continue
                rows.append(data)
                prev = data
    if not rows:
        raise SystemExit(f"no AZMET data found in {folder}")
    df = pd.DataFrame(rows, columns=UNIFIED_HEADERS)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Day of Year"] = pd.to_numeric(df["Day of Year"], errors="coerce").astype("Int64")
    df = df.sort_values(["Year", "Day of Year"]).reset_index(drop=True)
    for col in df.columns:
        if col not in META_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] == 999, col] = np.nan
    df = df[df["Year"].between(1900, 2100) & df["Day of Year"].between(1, 366)]
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + df["Day of Year"].astype(str).str.zfill(3),
                                format="%Y%j", errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    info["stations"] = sorted(df["Station Number"].astype(str).unique().tolist())
    value_cols = [c for c in df.columns if c not in META_COLS and c != "Date"]
    old_format = df["Year"] <= 2002
    absent = ["Reference ET", "Actual Vapor Pressure - Daily Mean", "Dewpoint - Daily Mean"]  # not in the 1987-2002 format
    info["missing_values_999"] = int(df.loc[~old_format, value_cols].isna().sum().sum()
                                     + df.loc[old_format, [c for c in value_cols if c not in absent]].isna().sum().sum())
    return df, info


def complete_calendar(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Insert empty rows for calendar days missing from the raw files (they are filled later)."""
    full = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    missing_days = full.difference(df["Date"])
    if len(missing_days):
        extra = pd.DataFrame({"Date": missing_days})
        extra["Year"] = extra["Date"].dt.year.astype("Int64")
        extra["Day of Year"] = extra["Date"].dt.dayofyear.astype("Int64")
        extra["Station Number"] = df["Station Number"].iloc[0]
        df = pd.concat([df, extra], ignore_index=True).sort_values("Date").reset_index(drop=True)
    return df, [str(d.date()) for d in missing_days]


def fill_soil_from_neighbours(stations: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    """Fill 20" and 4" soil-temperature gaps from the other stations (in place).

    A filled value is the mean of the neighbours' values of the same column on that day plus an offset
    interpolated linearly between the two gap ends (the last valid day before and the first after the gap).
    For the 20" sensor only the mean is filled this way; its Max and Min are the filled mean plus/minus the
    median (Max - Mean) and (Mean - Min) spreads of up to SPREAD_WINDOW_DAYS clean days on each side, so that
    a single unusual day at a gap end cannot widen the spread. The 4" Max, Mean and Min are each filled from
    the neighbours' same column. A neighbour is used only if it has a value on every day of the gap and at
    both ends (its own gaps of up to NEIGHBOUR_REF_INTERP_DAYS days are interpolated first). In hold-out
    tests on clean data this was more accurate than a straight line at every gap length for both sensors.
    Gaps without a usable neighbour or longer than SOIL20_FILL_MAX_DAYS are left to linear interpolation.
    """
    plans = [("20in", SOIL20[2], [SOIL20[2]]), ("4in", SOIL4[2], SOIL4)]  # (label, column defining the gap, columns filled)
    refs = {c: df.set_index("Date")[SOIL20 + SOIL4].astype(float).interpolate(limit=NEIGHBOUR_REF_INTERP_DAYS,
                                                                                limit_area="inside")
            for c, df in stations.items()}
    report: dict[str, list[str]] = {c: [] for c in stations}
    for c, df in stations.items():
        others = [o for o in stations if o != c]
        orig = df[SOIL20 + SOIL4].copy()
        for label, key, cols in plans:
            isna = df[key].isna() & (df["Date"].dt.year >= QC_FROM_YEAR)
            run_id = (isna != isna.shift()).cumsum()
            for _, idx in df[isna].groupby(run_id[isna]).groups.items():
                idx = list(idx)
                a, b = idx[0] - 1, idx[-1] + 1
                if a < 0 or b >= len(df) or len(idx) > SOIL20_FILL_MAX_DAYS:
                    continue
                span = list(range(a, b + 1))
                w = np.linspace(0.0, 1.0, len(span))
                used = set()
                for col in cols:
                    if pd.isna(orig[col].iloc[a]) or pd.isna(orig[col].iloc[b]):
                        continue
                    R = pd.concat([refs[o][col].reindex(df["Date"]).reset_index(drop=True) for o in others], axis=1)
                    ok = [k for k in range(R.shape[1]) if R.iloc[span, k].notna().all()]
                    if not ok:
                        continue
                    base = R.iloc[span, ok].mean(axis=1).to_numpy()
                    new = base + (1 - w) * (orig[col].iloc[a] - base[0]) + w * (orig[col].iloc[b] - base[-1])
                    df.loc[idx, col] = new[1:-1]
                    used.update(others[k] for k in ok)
                if not used:
                    continue
                if label == "20in":
                    s_max, s_min, s_mean = SOIL20
                    clean = orig[[s_max, s_mean, s_min]].notna().all(axis=1)
                    before = [i for i in range(max(0, a - 3 * SPREAD_WINDOW_DAYS), a + 1) if clean.iloc[i]][-SPREAD_WINDOW_DAYS:]
                    after = [i for i in range(b, min(len(df), b + 1 + 3 * SPREAD_WINDOW_DAYS)) if clean.iloc[i]][:SPREAD_WINDOW_DAYS]
                    if before and after:
                        for col, sign in ((s_max, 1.0), (s_min, -1.0)):
                            sp0 = float(np.median(sign * (orig[col].iloc[before] - orig[s_mean].iloc[before])))
                            sp1 = float(np.median(sign * (orig[col].iloc[after] - orig[s_mean].iloc[after])))
                            spread = (1 - w) * max(sp0, 0.0) + w * max(sp1, 0.0)
                            df.loc[idx, col] = (df.loc[span, s_mean].to_numpy() + sign * spread)[1:-1]
                    else:
                        df.loc[idx, [s_max, s_min]] = np.nan
                report[c].append(f"{label} {df.at[idx[0], 'Date'].date()}..{df.at[idx[-1], 'Date'].date()} "
                                 f"({len(idx)} days, from {' + '.join(sorted(used))})")
    return report


def enforce_ordering(df: pd.DataFrame) -> int:
    """Make Max >= Mean >= Min for every triple in TRIPLES on rows from QC_FROM_YEAR on (filled days and the few raw
    days whose mean lies outside their own max/min). Returns the number of cells changed."""
    n = 0
    for mx, me, mn in TRIPLES:
        rows = df["Date"].dt.year >= QC_FROM_YEAR
        hi = rows & (df[mx] < df[me])
        lo = rows & (df[mn] > df[me])
        df.loc[hi, mx] = df.loc[hi, me]
        df.loc[lo, mn] = df.loc[lo, me]
        n += int(hi.sum() + lo.sum())
    return n


def gap_summary(df: pd.DataFrame, cols: list[str]) -> dict:
    """Per column: number of missing days and the longest run of consecutive missing days."""
    gaps = {}
    for c in cols:
        isna = df[c].isna()
        if not isna.any():
            continue
        run_id = (isna != isna.shift()).cumsum()
        run_len = isna.groupby(run_id).sum()
        longest_id = run_len.idxmax()
        rows = df.index[run_id == longest_id]
        gaps[c] = {"missing_days": int(isna.sum()), "longest_run_days": int(run_len.max()),
                   "longest_run": f"{df.loc[rows[0], 'Date'].date()}..{df.loc[rows[-1], 'Date'].date()}"}
    return gaps


QC_FROM_YEAR = 2003  # the 1987-2002 file format has no 20" soil sensor (its 4" values sit in the 20" columns), so QC starts with the 2003 format


def qc_within_station(df: pd.DataFrame, county: str) -> tuple[pd.DataFrame, dict]:
    """Station-level plausibility rules (QC_RULES + MANUAL_EXCLUSIONS) on rows from QC_FROM_YEAR on."""
    df = df.copy()
    flags: dict[str, int] = {}
    details: dict[str, list[str]] = {}
    in_scope = df["Date"].dt.year >= QC_FROM_YEAR
    r = QC_RULES

    def flag(mask: pd.Series, cols: list[str], rule: str) -> None:
        mask = mask.fillna(False) & in_scope
        n = int(mask.sum())
        if n:
            df.loc[mask, cols] = np.nan
            flags[rule] = n
            d = df.loc[mask, "Date"].dt.date.astype(str).tolist()
            details[rule] = d[:6] + (["..."] if n > 6 else [])

    lo, hi = r["rh_clip"]
    for c in ("RH - Max", "RH - Min", "RH - Mean"):
        out = ((df[c] < lo) | (df[c] > hi)) & in_scope
        if out.any():
            rule = f"{c} clipped to [{lo:g},{hi:g}]"
            flags[rule] = int(out.sum())
            details[rule] = df.loc[out, "Date"].dt.date.astype(str).tolist()[:6] + (["..."] if out.sum() > 6 else [])
            df.loc[in_scope, c] = df.loc[in_scope, c].clip(lo, hi)

    s_max, s_min, s_mean = SOIL20
    rain = df["Precipitation - Total"].fillna(0.0)
    rainy = (rain >= r["rain_exempt_mm"]) | (rain.shift(1).fillna(0.0) >= r["rain_exempt_mm"])
    # the exemption is for storm cooling only: the 20" mean or min falls and the max does not jump up
    cooling = (df[s_mean] < df[s_mean].shift(1)) | (df[s_min] < df[s_min].shift(1))
    no_warm_jump = ~(df[s_max] > df[s_max].shift(1) + 1.0)
    rainy = rainy & cooling & no_warm_jump
    flag((df[s_max] - df[s_min] > r["soil20_daily_range_gt"]) & ~rainy, SOIL20,
         f"20in soil daily range > {r['soil20_daily_range_gt']:g} C (not on a cooling day on or after >= {r['rain_exempt_mm']:g} mm of rain)")
    flag(((df[s_max] - df[s_mean] > r["soil20_within_day_gt"]) | (df[s_mean] - df[s_min] > r["soil20_within_day_gt"])) & ~rainy,
         SOIL20, f"20in soil mean more than {r['soil20_within_day_gt']:g} C from the day's max or min (same rain exemption)")
    flag((df[SOIL20] < r["soil20_below"]).any(axis=1), SOIL20, f"20in soil below {r['soil20_below']:g} C")
    flag(df['4" Soil Temp - Max'] - df["Air Temp - Max"] > r["soil4max_minus_airmax_gt"], SOIL4,
         f"4in soil Max more than {r['soil4max_minus_airmax_gt']:g} C above air Max (all three 4in values)")
    flag(df['4" Soil Temp - Mean'] - df["Air Temp - Mean"] > r["soil4mean_minus_airmean_gt"], SOIL4,
         f"4in soil Mean more than {r['soil4mean_minus_airmean_gt']:g} C above air Mean (all three 4in values)")
    flag(df["Max Wind Speed"] > r["max_wind_speed_gt"], ["Max Wind Speed"], f"Max Wind Speed > {r['max_wind_speed_gt']:g} m/s")
    sd_low = df["Wind Direction Std Dev"] < r["wind_dir_sd_below"]
    fixed = df["Wind Vector Magnitude"] >= r["stuck_vane_ratio"] * df["Wind Speed - Mean"]
    core = (sd_low & fixed).fillna(False)
    stuck = core | (sd_low & (core.shift(1, fill_value=False) | core.shift(-1, fill_value=False)))
    flag(stuck, WIND_DIR, f"stuck vane (direction std dev < {r['wind_dir_sd_below']:g} and vector magnitude = mean speed)")
    for cty, cols, a, b, why in MANUAL_EXCLUSIONS:
        if cty == county:
            mask = (df["Date"] >= pd.Timestamp(a)) & (df["Date"] <= pd.Timestamp(b))
            flag(mask, cols, f"manual exclusion {a}..{b} ({why})")
    return df, {"flags": flags, "examples": details}


def qc_neighbours(stations: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Neighbour-adjusted jump and decay rules for the 20" soil mean (in place; see QC_RULES).

    The references are the other stations' 20" means after their station-level rules. A day is flagged only if
    it disagrees with every neighbour that has values on the compared days.
    """
    r = QC_RULES
    s_mean = SOIL20[2]
    refs = {c: df.set_index("Date")[s_mean] for c, df in stations.items()}
    result = {}
    for c, df in stations.items():
        others = [o for o in stations if o != c]
        x = df[s_mean].to_numpy(dtype=float)
        R = [refs[o].reindex(df["Date"]).to_numpy(dtype=float) for o in others]
        scope = (df["Date"].dt.year >= QC_FROM_YEAR).to_numpy()
        n = len(x)
        jump = np.zeros(n, dtype=bool)
        last = -1
        for i in range(n):
            if np.isnan(x[i]):
                continue
            if last >= 0 and scope[i] and i - last <= 5:
                devs = [abs((x[i] - x[last]) - (rr[i] - rr[last])) for rr in R
                        if not (np.isnan(rr[i]) or np.isnan(rr[last]))]
                if devs and min(devs) > r["soil20_jump_gt"]:
                    jump[i] = True
                    continue
            last = i
        xx = np.where(jump, np.nan, x)
        decay = np.zeros(n, dtype=bool)
        anchor, run, in_run, n_after = -1, 0, False, 0
        for i in range(n):
            if np.isnan(xx[i]):
                in_run, run, n_after = True, run + 1, 0
                continue
            if (in_run and scope[i] and anchor >= 0 and run <= r["soil20_decay_after_run_le"]
                    and n_after < r["soil20_decay_max_days"]):
                devs = [abs(xx[i] - (xx[anchor] + rr[i] - rr[anchor])) for rr in R
                        if not (np.isnan(rr[i]) or np.isnan(rr[anchor]))]
                if devs and min(devs) > r["soil20_decay_gt"]:
                    decay[i] = True
                    xx[i] = np.nan
                    n_after += 1
                    continue
            in_run, run, anchor = False, 0, i
        flags, examples = {}, {}
        for name, mask in ((f"20in soil jump > {r['soil20_jump_gt']:g} C against every neighbour station", jump),
                           (f"20in soil still more than {r['soil20_decay_gt']:g} C from the neighbour-adjusted pre-spike value "
                            f"(at most {r['soil20_decay_max_days']} days after a run of at most {r['soil20_decay_after_run_le']} flagged days)", decay)):
            if mask.any():
                df.loc[mask, SOIL20] = np.nan
                flags[name] = int(mask.sum())
                d = df.loc[mask, "Date"].dt.date.astype(str).tolist()
                examples[name] = d[:6] + (["..."] if len(d) > 6 else [])
        result[c] = {"flags": flags, "examples": examples}
    return result


def prepare_weather() -> dict[str, tuple[pd.DataFrame, dict]]:
    """Daily weather tables of all stations, ready for weekly aggregation.

    Missing calendar days inserted; station-level rules; neighbour rules for the 20" soil mean; 20" and 4" soil
    gaps filled from the neighbour stations; every other gap linearly interpolated; Max >= Mean >= Min on every day
    from 2003 on. All three stations are always processed because the neighbour steps need all of them.
    """
    stations: dict[str, pd.DataFrame] = {}
    infos: dict[str, dict] = {}
    for county, cfg in COUNTIES.items():
        info: dict = {"missing_calendar_days": [], "qc": {}, "neighbour_qc": {}, "soil20_fills": [], "ordering_fixes": 0}
        df, pinfo = parse_azmet_folder(cfg["azmet_dir"])
        info.update(pinfo)
        df = df.sort_values("Date").reset_index(drop=True)
        df, info["missing_calendar_days"] = complete_calendar(df)
        df, info["qc"] = qc_within_station(df, county)
        stations[county], infos[county] = df, info
    value_cols = [c for c in stations["Maricopa"].columns if c not in META_COLS and c != "Date"]
    for county, res in qc_neighbours(stations).items():
        infos[county]["neighbour_qc"] = res
    for county, df in stations.items():
        infos[county]["gaps"] = gap_summary(df, value_cols)
        infos[county]["missing_values_total"] = int(df[value_cols].isna().sum().sum())
    for county, filled in fill_soil_from_neighbours(stations).items():
        infos[county]["soil20_fills"] = filled
    out = {}
    for county, df in stations.items():
        df[value_cols] = df[value_cols].astype(float).interpolate(method="linear", limit_direction="both")
        infos[county]["ordering_fixes"] = enforce_ordering(df)
        df = add_mmwr(df)
        infos[county]["days"] = len(df)
        infos[county]["first_day"], infos[county]["last_day"] = df["Date"].min().date(), df["Date"].max().date()
        out[county] = (df, infos[county])
    return out


def weekly_weather(daily: pd.DataFrame) -> pd.DataFrame:
    """Mean over the days of each MMWR week of every numeric column (original logic)."""
    numeric = daily.select_dtypes(include="number").columns.difference(["MMWR_Year", "MMWR_Week"]).tolist()
    weekly = daily.groupby(["MMWR_Year", "MMWR_Week"])[numeric].mean().reset_index()
    weekly = weekly.drop(columns=[c for c in ("Day of Year", "Year") if c in weekly.columns])
    weekly["n_days"] = daily.groupby(["MMWR_Year", "MMWR_Week"]).size().values
    return weekly


# --------------------------------------------------------------------------------------------
# 2. EPA AirData PM10 -> daily monitor mean -> MMWR weekly means
# --------------------------------------------------------------------------------------------
def load_pm10_raw(folder: Path, pattern: str, county: str) -> tuple[pd.DataFrame, int]:
    files = sorted(glob.glob(str(folder / pattern)))
    if not files:
        raise SystemExit(f"no PM10 files matching {pattern} in {folder}")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d.columns = d.columns.str.strip()
        need = {"Date", "Daily Mean PM10 Concentration", "Daily AQI Value", "State", "County", "Site ID", "Method Code"}
        if not need.issubset(d.columns):
            raise SystemExit(f"{f}: missing columns {need - set(d.columns)}")
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw[(raw["State"] == "Arizona") & (raw["County"] == county)].copy()
    return raw, len(files)


def read_pm10_folder(folder: Path, pattern: str, county: str,
                     methods: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw, n_files = load_pm10_raw(folder, pattern, county)
    info = {"files": n_files, "rows": len(raw),
            "methods_in_data": sorted(int(m) for m in raw["Method Code"].dropna().unique()),
            "methods_kept": methods, "airnow_site_days": 0, "dropped_days_few_monitors": [],
            "imputed_days_few_monitors": [], "rule": ""}
    aqs = raw[raw["Method Code"].isin(methods)]
    info["monitors_selected"] = int(aqs["Site ID"].nunique())
    info["rows_selected"] = len(aqs)
    neg = raw["Daily Mean PM10 Concentration"] < 0
    info["negative_rows_clipped"] = int((neg & raw["Method Code"].isin(methods)).sum())
    raw = raw.copy()
    raw.loc[neg, "Daily Mean PM10 Concentration"] = 0.0
    aqs = raw[raw["Method Code"].isin(methods)]
    # per site-day first (collapses duplicate POCs), then over sites
    sd = aqs.groupby(["Date", "Site ID"], as_index=False)[PM10_COLS].mean()
    sd["source"] = "AQS"
    sites = set(sd["Site ID"])
    # a selected monitor keeps reporting through its preliminary AirNow rows once its quality-assured
    # AQS rows stop (interior AQS gaps are not filled: an invalidated day stays out). A site whose
    # AQS record continues under another method code after its last selected-method day changed
    # instrument and is not continued.
    last_sel = sd.groupby("Site ID")["Date"].max()
    last_any = raw[raw["Method Code"].notna()].groupby("Site ID")["Date"].max()
    recent = aqs["Date"].max() - pd.Timedelta(days=365)  # only monitors active within the last year of the AQS feed
    eligible = {s for s in sites if last_any.get(s) == last_sel[s] and last_sel[s] >= recent}
    an = raw[raw["Method Code"].isna() & raw["Site ID"].isin(eligible)]
    an = an[an["Date"] > an["Site ID"].map(last_sel)]
    info["airnow_sites"] = sorted(an["Site ID"].unique().tolist())
    an_sd = an.groupby(["Date", "Site ID"], as_index=False)[PM10_COLS].mean()
    an_sd["source"] = "AirNow"
    info["airnow_site_days"] = len(an_sd)
    if len(an_sd):
        info["airnow_first_day"], info["airnow_last_day"] = an_sd["Date"].min().date(), an_sd["Date"].max().date()
    sd = pd.concat([sd, an_sd], ignore_index=True)
    daily = sd.groupby("Date", as_index=False)[PM10_COLS].mean()
    daily["n_monitors"] = sd.groupby("Date")["Site ID"].nunique().values
    daily["n_airnow"] = sd.groupby("Date")["source"].apply(lambda s: int((s == "AirNow").sum())).values
    # minimum-monitor rule per calendar year
    year = daily["Date"].dt.year
    med = daily.groupby(year)["n_monitors"].median()
    thr = year.map(lambda y: max(2, math.ceil(PM10_MIN_MONITOR_FRAC * med[y])))
    drop = daily["n_monitors"] < thr
    # a day with too few monitors is estimated from the monitors that did report, scaled by their median
    # ratio to the full county mean over the previous year (days on which all of them reported and the
    # county mean passed the rule); without 30 such days it is dropped
    good = daily.loc[~drop].set_index("Date")[PM10_COLS]
    imputed = []
    for i in daily.index[drop]:
        day = daily.at[i, "Date"]
        today = sd[sd["Date"] == day]
        sites = set(today["Site ID"])
        hist = sd[(sd["Date"] < day) & (sd["Date"] >= day - pd.Timedelta(days=365)) & sd["Site ID"].isin(sites)]
        per_day = hist.groupby("Date")["Site ID"].nunique()
        days_all = per_day.index[per_day == len(sites)].intersection(good.index)
        if len(days_all) < 30:
            continue
        sub = hist[hist["Date"].isin(days_all)].groupby("Date")[PM10_COLS].mean()
        ratio = (good.loc[days_all, PM10_COLS] / sub[PM10_COLS].where(sub[PM10_COLS] > 0)).median()
        daily.loc[i, PM10_COLS] = (today[PM10_COLS].mean() * ratio)[PM10_COLS].to_numpy()
        drop.loc[i] = False
        imputed.append(f"{day.date()} ({len(sites)} of median {med[day.year]:g} monitors; their mean "
                       f"{today['Daily Mean PM10 Concentration'].mean():.1f} x ratio {ratio['Daily Mean PM10 Concentration']:.2f} "
                       f"= {daily.at[i, 'Daily Mean PM10 Concentration']:.1f} ug/m3)")
    info["imputed_days_few_monitors"] = imputed
    info["dropped_days_few_monitors"] = [f"{d.date()} ({n} of median {m:g})" for d, n, m in
                                        zip(daily.loc[drop, "Date"], daily.loc[drop, "n_monitors"], year[drop].map(med))]
    daily = daily[~drop].copy()
    info["rule"] = (f"mean per monitor-day, then over monitors; a selected monitor continues through its AirNow rows after "
                    f"its last AQS day; a day with fewer than max(2, ceil({PM10_MIN_MONITOR_FRAC} x yearly median)) monitors "
                    f"is the reporting monitors' mean times their median ratio to the county mean over the previous year")
    daily["Daily Mean PM10 Concentration"] = daily["Daily Mean PM10 Concentration"].round(2)
    daily["Daily AQI Value"] = daily["Daily AQI Value"].round(1)
    daily = daily.sort_values("Date").reset_index(drop=True)
    info["first_day"], info["last_day"] = daily["Date"].min().date(), daily["Date"].max().date()
    info["last_aqs_day"] = aqs["Date"].max().date()
    full = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
    info["missing_days"] = int(len(full) - len(daily))
    daily = add_mmwr(daily)
    weekly = daily.groupby(["MMWR_Year", "MMWR_Week"])[PM10_COLS].mean().reset_index()
    weekly["n_days"] = daily.groupby(["MMWR_Year", "MMWR_Week"]).size().values
    weekly["monitors_mean"] = daily.groupby(["MMWR_Year", "MMWR_Week"])["n_monitors"].mean().round(1).values
    return daily, weekly, info


# --------------------------------------------------------------------------------------------
# 3. ADHS county workbooks -> weekly cases
# --------------------------------------------------------------------------------------------
def read_case_workbooks(paths: list[Path]) -> tuple[pd.DataFrame, dict]:
    """Weekly cases by county, MMWR year from the sheet name; later workbooks override years."""
    by_year: dict[int, pd.DataFrame] = {}
    provenance: dict[int, str] = {}
    for path in paths:
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            m = re.search(r"(\d{4})", sheet)
            if not m or "intro" in sheet.lower():
                continue
            year = int(m.group(1))
            d = xl.parse(sheet, header=None)
            header_row = None
            for i in range(min(4, len(d))):
                vals = [str(v).strip().upper() for v in d.iloc[i].tolist()]
                if "MARICOPA" in vals and "PIMA" in vals:
                    header_row = i
                    break
            if header_row is None:
                raise SystemExit(f"{path.name} / {sheet}: no header row with county names")
            cols = [str(v).strip().upper() for v in d.iloc[header_row].tolist()]
            body = d.iloc[header_row + 1:].copy()
            body.columns = cols
            week = pd.to_numeric(body.iloc[:, 0], errors="coerce")
            body = body[week.notna()].copy()
            body["MMWR_Week"] = week[week.notna()].astype(int).values
            table = pd.DataFrame({"MMWR_Year": year, "MMWR_Week": body["MMWR_Week"].values})
            for c in ("MARICOPA", "PIMA", "PINAL", "TOTAL"):
                table[c] = pd.to_numeric(body[c], errors="coerce").values
            if table[["MARICOPA", "PIMA", "PINAL"]].isna().any().any():
                raise SystemExit(f"{path.name} / {sheet}: empty county cells")
            if table["MMWR_Week"].duplicated().any():
                raise SystemExit(f"{path.name} / {sheet}: duplicated weeks")
            if table["MMWR_Week"].max() > weeks_in_mmwr_year(year):
                raise SystemExit(f"{path.name} / {sheet}: week {table['MMWR_Week'].max()} does not exist in MMWR year {year}")
            by_year[year] = table.sort_values("MMWR_Week").reset_index(drop=True)
            provenance[year] = f"{path.name} / {sheet} (weeks {table.MMWR_Week.min()}-{table.MMWR_Week.max()})"
    cases = pd.concat([by_year[y] for y in sorted(by_year)], ignore_index=True)
    for c in ("MARICOPA", "PIMA", "PINAL"):
        cases[c] = cases[c].astype(int)
    dropped = None
    if len(cases) > 9:
        last_total = cases["TOTAL"].iloc[-1]
        ref = cases["TOTAL"].iloc[-9:-1].median()
        if pd.notna(last_total) and last_total < CASES_PARTIAL_WEEK_FRAC * ref:
            dropped = f"{week_label(cases.MMWR_Year.iloc[-1], cases.MMWR_Week.iloc[-1])} (statewide total {last_total:g} vs median {ref:g} of the previous 8 weeks)"
            cases = cases.iloc[:-1].copy()
    return cases, {"provenance": provenance, "dropped_partial_week": dropped}


# --------------------------------------------------------------------------------------------
# 4. merge
# --------------------------------------------------------------------------------------------
def assemble(county: str, cases: pd.DataFrame, weather: pd.DataFrame, pm10: pd.DataFrame,
             end: tuple[int, int] | None) -> tuple[pd.DataFrame, dict]:
    col = COUNTIES[county]["cases_col"]
    first = (2006, 1)
    last_cases = tuple(cases[["MMWR_Year", "MMWR_Week"]].iloc[-1].tolist())
    last = min(last_cases, end) if end else last_cases
    cal = mmwr_calendar(first, last)
    w = weather.rename(columns={"Reference ET (Penman-Monteith)": "Reference ET Penman-Monteith"})
    out = (cal.merge(cases[["MMWR_Year", "MMWR_Week", col]], how="left")
              .merge(w[["MMWR_Year", "MMWR_Week"] + FINAL_WEATHER_COLS + ["n_days"]], how="left")
              .merge(pm10[["MMWR_Year", "MMWR_Week"] + PM10_COLS + ["n_days"]].rename(columns={"n_days": "n_days_pm10"}), how="left"))
    out.insert(0, "Date", [week_label(y, k) for y, k in zip(out["MMWR_Year"], out["MMWR_Week"])])
    partial = {"weather": out.loc[out["n_days"].fillna(0) < 7, ["Date", "n_days"]].values.tolist(),
               "pm10": out.loc[out["n_days_pm10"].fillna(0) < 7, ["Date", "n_days_pm10"]].values.tolist()}
    out = out.drop(columns=["MMWR_Year", "MMWR_Week", "n_days", "n_days_pm10"])
    out = out[["Date", col] + FINAL_WEATHER_COLS + PM10_COLS]
    filled = {}
    for c in FINAL_WEATHER_COLS + PM10_COLS:
        if out[c].isna().any():
            filled[c] = out.loc[out[c].isna(), "Date"].tolist()
            out[c] = out[c].interpolate(method="linear", limit_direction="both")
    missing = {c: int(out[c].isna().sum()) for c in out.columns if out[c].isna().any()}
    info = {"first": out["Date"].iloc[0], "last": out["Date"].iloc[-1], "rows": len(out),
            "missing_cells": missing, "partial_weeks": partial, "weekly_gaps_filled": filled}
    return out, info


# --------------------------------------------------------------------------------------------
def parse_week(s: str | None) -> tuple[int, int] | None:
    if not s:
        return None
    y, w = s.split("_")
    return int(y), int(w)


def build_county(county: str, cases: pd.DataFrame, end, out: Path,
                 weather_entry: tuple[pd.DataFrame, dict]) -> tuple[pd.DataFrame, list[str], tuple]:
    cfg = COUNTIES[county]
    out.mkdir(parents=True, exist_ok=True)
    log(f"\n=== {county} ===")
    report = [f"## {county}"]
    daily, winfo = weather_entry
    weather = weekly_weather(daily)
    log(f"  weather: {winfo['files']} files, {winfo['days']} days {winfo['first_day']}..{winfo['last_day']}, station(s) {winfo['stations']}, "
        f"skipped lines {winfo['skipped_lines']}, duplicate lines {winfo['duplicate_lines']}, 999 values {winfo['missing_values_999']}, "
        f"missing calendar days {len(winfo['missing_calendar_days'])} {winfo['missing_calendar_days'][:5]}")
    if winfo["qc"].get("flags"):
        log(f"  QC flags (station rules): {winfo['qc']['flags']}")
    if winfo["neighbour_qc"].get("flags"):
        log(f"  QC flags (neighbour rules): {winfo['neighbour_qc']['flags']}")
    if winfo["soil20_fills"]:
        log(f"  soil gaps filled from neighbour stations: {len(winfo['soil20_fills'])} ({winfo['soil20_fills'][:3]} ...)")
    long_gaps = {c: g for c, g in winfo["gaps"].items() if g["longest_run_days"] >= 7 and c not in ("Reference ET", "Actual Vapor Pressure - Daily Mean", "Dewpoint - Daily Mean")}
    if long_gaps:
        log(f"  gaps of a week or more (filled): {long_gaps}")
    save_csv(daily, out / f"{county}_weather_daily.csv")
    save_csv(weather, out / f"{county}_weather_weekly_mmwr.csv")

    methods = cfg["pm10_methods"]
    pm10_daily, pm10_weekly, pinfo = read_pm10_folder(cfg["pm10_dir"], cfg["pm10_glob"], cfg["pm10_county"], methods)
    log(f"  PM10: {pinfo['files']} files, methods in data {pinfo['methods_in_data']}, kept {methods} ({pinfo['monitors_selected']} monitors, "
        f"{pinfo['rows_selected']}/{pinfo['rows']} rows); AQS days to {pinfo['last_aqs_day']}; AirNow site-days used {pinfo['airnow_site_days']}"
        + (f" ({pinfo.get('airnow_first_day')}..{pinfo.get('airnow_last_day')})" if pinfo["airnow_site_days"] else "")
        + f"; days with too few monitors: estimated {len(pinfo['imputed_days_few_monitors'])} {pinfo['imputed_days_few_monitors'][:3]}, "
        f"dropped {len(pinfo['dropped_days_few_monitors'])} {pinfo['dropped_days_few_monitors'][:4]}; "
        f"days without a value inside the range: {pinfo['missing_days']}")
    save_csv(pm10_daily, out / f"{county}_pm10_daily.csv")
    save_csv(pm10_weekly, out / f"{county}_pm10_weekly_mmwr.csv")

    final, finfo = assemble(county, cases, weather, pm10_weekly, end)
    name = f"Clean Processed Data2_{county}.csv"
    written = save_csv(final, out / name)
    log(f"  -> {name}: {finfo['rows']} rows {finfo['first']}..{finfo['last']}, missing cells {finfo['missing_cells'] or 'none'}; "
        f"weeks with fewer than 7 weather days {len(finfo['partial_weeks']['weather'])}, PM10 days {len(finfo['partial_weeks']['pm10'])}; "
        f"weekly gaps filled {({k: len(v) for k, v in finfo['weekly_gaps_filled'].items()}) or 'none'}")

    gap_text = "; ".join(f"{c}: {g['missing_days']} days, longest {g['longest_run_days']} ({g['longest_run']})"
                         for c, g in winfo["gaps"].items() if g["longest_run_days"] >= 3
                         and c not in ("Reference ET", "Actual Vapor Pressure - Daily Mean", "Dewpoint - Daily Mean")) or "none of 3 days or more"
    def _qc_text(q: dict) -> str:
        return "; ".join(f"{k}: {v} days (e.g. {', '.join(q['examples'].get(k, [])[:3])})" for k, v in q.get("flags", {}).items()) or "none"
    qc_text = _qc_text(winfo["qc"])
    nb_text = _qc_text(winfo["neighbour_qc"])
    report += [
        f"weather: {winfo['files']} AZMET files, {winfo['days']} days {winfo['first_day']}..{winfo['last_day']}, station(s) {winfo['stations']}; "
        f"skipped lines {winfo['skipped_lines']}, consecutive duplicates removed {winfo['duplicate_lines']}, 999 values {winfo['missing_values_999']}, "
        f"missing calendar days {len(winfo['missing_calendar_days'])} {winfo['missing_calendar_days'][:5]}",
        f"weather QC, station rules (values set to missing; RH only clipped): {qc_text}",
        f"weather QC, neighbour rules for the 20in soil mean: {nb_text}",
        f"soil-temperature gaps filled from the neighbour stations: {len(winfo['soil20_fills'])} {winfo['soil20_fills'] or ''}",
        f"Max >= Mean >= Min enforced (days from {QC_FROM_YEAR} on): {winfo['ordering_fixes']} cells changed",
        f"PM10 rows with negative concentration set to 0: {pinfo['negative_rows_clipped']}",
        f"PM10 days with too few monitors estimated from the reporting monitors: {pinfo['imputed_days_few_monitors'] or 'none'}",
        f"weather gaps after QC, before filling: {gap_text}",
        f"PM10: methods in data {pinfo['methods_in_data']}, kept {methods} ({pinfo['monitors_selected']} monitors); rule: {pinfo['rule']}; "
        f"AQS rows to {pinfo['last_aqs_day']}; AirNow site-days used {pinfo['airnow_site_days']}; "
        f"days dropped for too few monitors: {len(pinfo['dropped_days_few_monitors'])} {pinfo['dropped_days_few_monitors'][:6]}; "
        f"days without a value: {pinfo['missing_days']}",
        f"final: {name}, {finfo['rows']} rows {finfo['first']}..{finfo['last']}, missing cells {finfo['missing_cells'] or 'none'}",
        f"weeks built from fewer than 7 weather days: {finfo['partial_weeks']['weather'] or 'none'}",
        f"weeks built from fewer than 7 PM10 days: {len(finfo['partial_weeks']['pm10'])} {finfo['partial_weeks']['pm10']}",
        f"weekly values interpolated because a whole week was missing: {finfo['weekly_gaps_filled'] or 'none'}",
    ]
    return final, report, (weather, pm10_daily, pm10_weekly)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--counties", nargs="+", default=list(COUNTIES), choices=list(COUNTIES))
    p.add_argument("--end", default=DEFAULT_END,
                   help="last MMWR week to keep (default: %(default)s, so that all counties cover the same weeks; "
                        "'cases' = last complete week with case counts)")
    p.add_argument("--out", default=None, help="output folder (default: output/)")
    args = p.parse_args(argv)
    out = Path(args.out) if args.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    end = None if args.end == "cases" else parse_week(args.end)

    self_test_mmwr()
    log("MMWR week function: self-test passed (1985-2030)")

    cases, cinfo = read_case_workbooks(CASE_WORKBOOKS)
    cases["Date"] = [week_label(y, w) for y, w in zip(cases["MMWR_Year"], cases["MMWR_Week"])]
    save_csv(cases[["Date", "MMWR_Year", "MMWR_Week", "MARICOPA", "PIMA", "PINAL", "TOTAL"]], out / "cases_weekly_by_county.csv")
    log(f"cases: {cases.Date.iloc[0]} .. {cases.Date.iloc[-1]} ({len(cases)} weeks); dropped partial week: {cinfo['dropped_partial_week']}")
    report = ["# Preprocessing report", "",
              f"cases: {cases.Date.iloc[0]} .. {cases.Date.iloc[-1]} ({len(cases)} weeks); trailing partial week dropped: {cinfo['dropped_partial_week']}"]
    report += [f"  {y}: {cinfo['provenance'][y]}" for y in sorted(cinfo["provenance"])]
    report.append("  (the most recent weeks of the newest workbook are preliminary: ADHS revises them in later workbooks)")

    weather = prepare_weather()
    for county in args.counties:
        final, rep, parts = build_county(county, cases, end, out, weather[county])
        report += [""] + rep
    save_text("\n".join(report) + "\n", out / "preprocessing_report.md")
    log(f"\nreport: {out / 'preprocessing_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
