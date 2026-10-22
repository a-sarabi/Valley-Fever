# Preprocessing

`preprocess_counties.py` builds the weekly files in `Data/Processed Data/` from the original data in this repository,
with the same steps for Maricopa, Pima and Pinal. Each file has the layout

```
Date, <COUNTY cases>, 25 AZMET weather columns (alphabetical), Daily AQI Value, Daily Mean PM10 Concentration
```

with 1083 rows (MMWR weeks 2006_1 .. 2026_39) and no missing cells.

## Run

Requires Python 3 with pandas, numpy and openpyxl (tested with Python 3.13, pandas 2.2, numpy 2.2 and openpyxl 3.1).
The `output/` folder is not tracked by git.

```
cd Data/Preprocessing
python preprocess_counties.py                                     # all counties -> output/
python preprocess_counties.py --counties Pima Pinal --end 2026_39
```

The weather of all three stations is always read, because the cleaning rules compare each station with the other
two. If an output file is open in Excel, the script writes `<name>.new.csv` (or `.new.md`) instead and says so.
The program writes these files to `output/`:

| output | content |
|---|---|
| `Clean Processed Data2_<County>.csv` | final weekly files, identical to those in `Data/Processed Data/` |
| `<County>_weather_daily.csv`, `<County>_weather_weekly_mmwr.csv` | daily AZMET table after cleaning and filling, and its MMWR-week means (`n_days` = days in the week) |
| `<County>_pm10_daily.csv`, `<County>_pm10_weekly_mmwr.csv` | monitor-averaged daily PM10/AQI (`n_monitors`, `n_airnow`) and MMWR-week means |
| `cases_weekly_by_county.csv` | weekly cases for MARICOPA, PIMA, PINAL and the statewide total |
| `preprocessing_report.md` | what was read, flagged, filled, estimated or dropped, per county |

## Steps

1. **Weather.** Every `SSYYrd.txt` AZMET file (station `SS`, year `YY`) is parsed: non-printable characters
   removed, lines with the wrong number of fields skipped (28 fields from 2003 on, 25 before), consecutive duplicate
   lines dropped, `999` set to missing, dates from year and day-of-year, MMWR year/week assigned. The daily values are
   then cleaned and filled (below), and the mean over the days of each MMWR week is taken for every variable,
   precipitation included.
2. **PM10 / AQI.** EPA AirData daily files are filtered to the county and to the monitor method codes below,
   averaged per monitor-day and then over monitors (PM10 rounded to 2 decimals, AQI to 1), and over the days of each
   MMWR week.
3. **Cases.** The ADHS "MMWR by County" workbooks are read sheet by sheet (MMWR year from the sheet name, header row
   found by the county names, `Total` rows ignored, week numbers checked against the MMWR calendar). A newer workbook
   overrides an older one for the same year: 2006-2023 from `MMWR by County (2006-2024).xlsx`, 2024 from
   `(2024-2025_ Updated)`, 2025-2026 from `(2025-26_ MMWRWK-YR)`.
4. **Merge** on MMWR year and week from 2006_1 to 2026_39 (option `--end`), the last complete week of case counts,
   so that all counties cover the same weeks.

## Cleaning rules

**Weather.** The raw files contain physically impossible readings, for example the Coolidge 20" soil sensor failing
for 2.5 months in 2021 (readings 5-6 C low before it went to 999); winter 20" spikes with multi-day decays at
Coolidge, to 21-24 C in 2017 (2017-01-28, from about 13 C) and to 29-33 C in 2019 (2019-02-18/19 and
2019-03-09/10), with the 4" sensor spiking on 2019-02-18..20; negative 20" values and 12 days whose 20" Min was 999
at Tucson in winter 2016-17, four of them (2017-02-05, 02-06, 02-07, 02-14) with corrupted means of 4.5, -9.6, 2.7
and 8.7 C; a wind vane stuck for 17 days (Coolidge, December 2018); a 61.6 m/s wind gust (Maricopa, 2019-04-17); 4"
soil readings of 48-60 C on rainy days (Tucson, July 2006); and weekly watering spikes of the 20" sensor at the
Maricopa station (most of all in 2026). These rules set such values to missing:

* **Station rules** (`QC_RULES`, from 2003 on, since the older file format has no 20" sensor): 20" daily
  range > 4 C, or 20" mean more than 4 C from the day's max or min, both not applied when the soil cools on a day with at
  least 10 mm of rain on it or the day before (real storm cooling, e.g. Maricopa 2008-05-22 and 2018-10-02); any 20"
  value < 0 C; 4" Max more than 20 C above the air Max, or 4" Mean more than 12 C above the air Mean (all three 4"
  values); Max Wind Speed > 35 m/s; stuck vane = direction std dev < 15 **and** vector magnitude equal to the mean
  speed (or an adjoining such day; real steady east-wind days are kept); RH clipped to 0-100.
* **Neighbour rules** for the 20" mean, which compare each station with each of the other two and flag a day only if
  it disagrees with all of them, so that a regional cold front or a fault at one neighbour flags nothing: a change
  since the last valid day that differs by more than 3 C from the neighbour's change; and, after a flagged run of at
  most 10 days, every following day (at most 21) that is still more than 1.5 C from the last clean pre-run value
  moved by the neighbour's change (the decay of a spike).
* **Documented failures** (`MANUAL_EXCLUSIONS`): Coolidge 20" soil 2021-08-24 .. 2021-11-13; Tucson 4" soil
  2006-07-27 .. 07-31; Coolidge 4" and 20" soil 2019-02-17 .. 02-21 (the spike with its onset and tail days, which
  the generic rules leave in and which would otherwise anchor the fill).

**Filling.** Missing daily values and missing calendar days are filled. Soil-temperature gaps of up to 120 days are
filled from the neighbour stations: the mean of the neighbours' values of the same column plus an offset
interpolated between the two gap ends. For the 20" sensor only the mean is filled this way and its Max and Min are
the filled mean plus/minus the median spread of up to 7 clean days on each side of the gap; the 4" Max, Mean and Min
are each filled from the neighbours' same column. In hold-out tests on clean data this was the most accurate method
at every gap length: for the 20" mean, root-mean-square errors of 0.21, 0.34, 0.59 and 0.90 C for gaps of 3, 7, 23
and 82 days, against 0.25, 0.45, 0.81 and 1.71 C for a straight line and 0.29-1.04 C for the station's own 4"
series; for the 4" mean, 0.72, 0.96, 1.11 and 1.37 C for gaps of 3, 5, 7 and 23 days against 0.86, 1.14, 1.37 and
1.88 C for a straight line. All other gaps, and soil gaps that are longer or have no neighbour with data on every
day of the gap, are interpolated linearly. Afterwards Max >= Mean >= Min is enforced on every day from 2003 on (5
cells: two Tucson days of 2017 whose 20" Min was 999 and was interpolated above the measured mean, and three raw days
whose mean lay outside their own max or min).

**PM10.**

* Monitors with two collocated instruments (POCs) are averaged per monitor-day first, then over monitors, so that each
  monitor counts once. Negative daily readings (2 rows) are set to 0.
* Quality-assured (AQS) rows stop before the end of the data (Maricopa: all but one monitor on 2026-06-30; Pima and
  Pinal: all monitors); the remaining months exist only as preliminary AirNow rows without a method code. A selected
  monitor continues through its AirNow rows after its last AQS day, provided it was active within the last year of the
  AQS feed and its AQS record did not continue under another instrument (such sites changed instrument and are not
  continued). Interior AQS gaps are not filled.
* A day with fewer than max(2, ceil(0.3 x that year's median monitor count)) monitors is estimated as the reporting
  monitors' mean times their median ratio to the county mean over the previous year (days on which all of them
  reported); without 30 such days it is dropped. This happened once: Pinal 2026-07-12, 2 of 11 monitors during a
  regional dust event, 141.5 x 1.13 = 160.3 ug/m3.
* Pima: method code 79 (TEOM) exists in Pima only in 2007-2015, and the Met One BAM monitors (122) cover every year but
  are only the Green Valley monitors until 2012-09-08 (one site, plus a second Green Valley site on 351 days from
  2007-10-12 to 2008-10-24), so a single method would give an artificial level shift when Tucson monitors switch to
  BAM. Pima therefore uses the method codes 79 (TEOM), 122 (BAM) and 127 (Partisol filter samplers, until 2017): 2-8
  monitors per day (2 on 32 days, never 1), identical to 122-only from 2018 on.

**Cases.** A trailing week whose statewide total is below half the median of the previous 8 weeks is dropped
(2026_40: statewide total 15 against about 250). Counts of the whole current year are revised in later workbooks:
every week of 2024 and 20 of 21 weeks of 2025 changed between workbooks, by up to -9% for Maricopa (-7.5% for the
statewide total); the small Pima and Pinal weekly counts changed by -13% to +27% and -15% to +33% in 2024 (Pinal down
to -12.5% in 2025). The largest statewide and Maricopa revisions, apart from a partial last week, are in the last two
months.

Settings per county:

| county | AZMET station | PM10 monitors kept |
|---|---|---|
| Maricopa | 6 (Maricopa Agricultural Center) | method 79, 19 monitors |
| Pima | 1 (Tucson) | methods 79, 122, 127, 9 monitors |
| Pinal | 5 (Coolidge) | method 79, 13 monitors |

## Known limitations (left in the data)

* **Sensor offsets that the rules do not homogenise.** The Coolidge 20" sensor reads 2.8 C above Maricopa's in
  2006-2016, 5.3 C above it between its 2017 fault and its 2021 failure (2017-02-20 .. 2021-08-23), and 2.2 C above it
  after its replacement on 2021-11-14; within a season the offset also drifts (about 5 to 9 C against the neighbours'
  mean from February to April 2019), so the residual of a decaying spike inside that range cannot be told apart from
  a real difference. The Tucson 20" sensor reads 3.9 C below Maricopa's until 2021 and 1.1 C below it from 2022. The
  Coolidge 4" sensor reads low in 2015-2017 until a step of about +6.5 C (+6.4 to +7.0 C against the neighbour
  stations) on 2017-03-03/04.
* **PM10 network composition.** The county mean moves when monitors are added or removed: Pinal 2006-2015 is dominated
  by the Cowtown feedlot monitor; the Maricopa TEOM network grew from 5 monitors in 2006 (about 6 per day through 2009,
  8 in 2010) to 13; in Pima, Ajo's switch from TEOM to BAM in May 2015 lowers the county mean by about 2 ug/m3, and
  Corona de Tucson and Tangerine joining the BAM set on 2025-04-01 (they reported under method 126 before) lowers it
  by about 2.2 ug/m3 (-9%).
* **Weekly means.** The weekly wind direction is an arithmetic mean of a circular variable, and precipitation is
  averaged, not summed, over the week.

## Checks built in

* The MMWR week function is self-tested against an independent implementation of the CDC rule for every day
  1985-2030 (MMWR years 2008, 2014, 2020 and 2025 have 53 weeks within the data range).
* `preprocessing_report.md` lists, per county, the raw-file statistics, every station and neighbour QC flag with
  example dates, the total missing days of each column and its longest gap when that gap is three days or more,
  every soil-temperature gap filled from the neighbours, the PM10 rule and monitor counts, PM10 days estimated or
  dropped, and every week built from fewer than seven weather or PM10 days.
