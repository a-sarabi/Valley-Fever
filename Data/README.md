# Data

Weekly Valley fever cases, weather and PM10 (dust) data for three Arizona counties: Maricopa, Pima and Pinal. Weeks are MMWR weeks, the CDC's epidemiological weeks, which run from Sunday to Saturday.

This folder has the original data (the weather and PM10 files as downloaded, and the weekly case counts from ADHS) and the processed weekly files made from them. The original data remain under the terms of their sources (ADHS, AZMET and the U.S. EPA); the MIT license of this repository does not apply to them. The processed files are derived from these data and are not products of ADHS, AZMET or the EPA.

## Processed data

`Processed Data/` has one file per county. The program in `Preprocessing/` makes them from the original data and reproduces them exactly (see `Preprocessing/README.md`):

| File | Cases | AZMET weather station | EPA PM10 monitors used |
|---|---|---|---|
| `Clean Processed Data2_Maricopa.csv` | Maricopa County | Maricopa Agricultural Center (station 6) | 19 monitors (method code 79) |
| `Clean Processed Data2_Pima.csv` | Pima County | Tucson (station 1) | 9 monitors (method codes 79, 122 and 127) |
| `Clean Processed Data2_Pinal.csv` | Pinal County | Coolidge (station 5) | 13 monitors (method code 79) |

The Maricopa Agricultural Center station is in Pinal County. Monitor counts cover 2006 to 2026; fewer monitors report on any one day.

- Rows: MMWR weeks 2006_1 to 2026_39 (1,083 weeks, 1 January 2006 to 3 October 2026). `Date` is the MMWR year and week, for example `2006_1`.
- Columns: `Date`, the county's weekly number of cases (`MARICOPA`, `PIMA` or `PINAL`), 25 weather variables from AZMET, `Daily AQI Value` and `Daily Mean PM10 Concentration`. No values are missing.
- Weather, PM10 and AQI values are means of the seven daily values of the week, also in columns whose names say "Daily" or "Total". `Precipitation - Total` is therefore the average daily total, not the weekly sum, and `Wind Vector Direction` is a simple average of compass directions (350° and 10° average to 180°).
- Units: temperatures and dew point in °C (`4"` and `20"` are soil depths in inches); relative humidity in %; vapor pressure and vapor pressure deficit (VPD) in kPa; solar radiation in MJ/m²; precipitation and reference evapotranspiration (ET) in mm; wind speeds in m/s; wind directions in degrees; heat units in °C (thresholds 12.8 and 30 °C); PM10 in µg/m³. Solar radiation, precipitation, ET and heat units are average daily amounts. The AQI (EPA's Air Quality Index) has no unit.
- Weather: daily values that fail quality checks, such as readings of failed or spiking soil sensors, are removed, and relative humidity above 100% is set to 100%. Removed and missing values are then filled: soil temperatures from the other stations (gaps of up to 120 days), everything else by linear interpolation.
- PM10 and AQI: daily values are averaged over each monitor's instruments, then over the county's monitors, then over the week. From July 2026 they come mostly from EPA's preliminary AirNow data, which may still change.
- Cases: each year comes from the newest ADHS workbook that covers it. ADHS revises the counts of the current year in later workbooks, the latest weeks most, so the 2026 counts are preliminary.
- Not corrected: the county PM10 mean shifts when monitors join, leave or change instrument, and the soil temperature sensors of the three stations read several degrees apart, by amounts that change over time (for example at Coolidge in 2017 and 2021).

## Original data

- Cases: weekly counts for each county (`MARICOPA`, `PIMA`, `PINAL`) and for Arizona (`TOTAL`), in `Valley Fever Data/cases_weekly_by_county.csv`, taken from the "MMWR by County" workbooks of the Arizona Department of Health Services (ADHS), which are not included here.
- Weather: daily files of the Arizona Meteorological Network (AZMET), University of Arizona, one station per county:
  - Maricopa: Maricopa Agricultural Center, station 6, <https://azmet.arizona.edu/azmet/06.htm>, in `Maricopa/Weather Data MARICOPA/Original Data MARICOPA/`
  - Pima: Tucson, station 1, <https://azmet.arizona.edu/azmet/01.htm>, in `Pima (Tucson)/Weather Data Pima/Original Data Pima/`
  - Pinal: Coolidge, station 5, <https://azmet.arizona.edu/azmet/05.htm>, in `Pinal (Coolidge)/Weather Data Coolidge/Original Data Coolidge/`

  The files are named `SSYYrd.txt` (station `SS`, year `YY`) and have one line per day; four files repeat one day on an identical line, which the processing counts once. The fields are listed at <https://azmet.arizona.edu/azmet/raw2003.htm> for files from 2003 on and at <https://azmet.arizona.edu/azmet/raw1987.htm> for the Maricopa files of 1987 to 2002.
- PM10 and AQI: daily data of the U.S. Environmental Protection Agency (EPA), Outdoor Air Quality Data, Download Daily Data, <https://www.epa.gov/outdoor-air-quality-data/download-daily-data>, in `Maricopa/PM10 Data/`, `Pima (Tucson)/PM10 Data/` and `Pinal (Coolidge)/PM10 Data/`. The files are named `<County> <Year>.csv` and list the county's PM10 monitors, not only those used here, with one row per monitor (site), instrument (`POC`) and day. Rows whose `Source` is `AirNow` are preliminary; nearly all of them are from 2026.

## Note on 2024

ADHS estimates that up to 800 of Arizona's 2024 cases were false positives from a faulty batch of test kits used from January to March 2024. These cases are included in the weekly counts. See the [ADHS 2024 Valley fever report](https://www.azdhs.gov/documents/preparedness/epidemiology-disease-control/valley-fever/reports/valley-fever-2024.pdf).
