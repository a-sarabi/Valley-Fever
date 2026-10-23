# Valley Fever, Weather and PM10 Data for Three Arizona Counties

[![Rebuild processed data](https://github.com/a-sarabi/Valley-Fever/actions/workflows/reproduce.yml/badge.svg)](https://github.com/a-sarabi/Valley-Fever/actions/workflows/reproduce.yml)

Weekly Valley fever (coccidioidomycosis) case counts for Maricopa, Pima and Pinal counties, Arizona, with weekly weather and PM10 (dust) data. The data cover MMWR weeks 2006_1 to 2026_39 (1 January 2006 to 3 October 2026).

## Contents

- `Data/Processed Data/`: the final weekly files, one per county.
- `Data/Maricopa/`, `Data/Pima (Tucson)/` and `Data/Pinal (Coolidge)/`: the original daily weather files (AZMET) and PM10 files (EPA) for each county.
- `Data/Valley Fever Data/`: the weekly case counts for each county (ADHS).
- `Data/Preprocessing/`: the program that builds the processed files from the original data.

[Data/README.md](Data/README.md) describes the sources, columns, units and processing.

## Using the data

Each processed file has one row per MMWR week, with the county's weekly number of cases, 25 weather variables, PM10 and the air quality index (AQI). For example, in Python, started in the repository's top folder:

```python
import pandas as pd

pima = pd.read_csv("Data/Processed Data/Clean Processed Data2_Pima.csv")
```

## Rebuilding the processed files

The program in `Data/Preprocessing/` rebuilds the processed files from the original data and checks that they are
identical, byte for byte, to those in `Data/Processed Data/`. You need Python 3.9 or newer. Run the commands below in
the repository's top folder, the one that contains `requirements.txt` (after Download ZIP it is called
`Valley-Fever-main`). They install numpy and pandas in a separate environment, `.venv`, so nothing else on your
computer changes.

macOS and Linux:

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd Data/Preprocessing
python preprocess_counties.py --check
```

Windows (PowerShell or Command Prompt):

```
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
cd Data\Preprocessing
python preprocess_counties.py --check
```

A successful run ends with three lines such as
`Clean Processed Data2_Pima.csv: identical to Data/Processed Data/Clean Processed Data2_Pima.csv`. With `--check`, the
program stops with an error (exit status 1) if a file differs. If PowerShell says that running scripts is disabled,
run `Set-ExecutionPolicy -Scope Process RemoteSigned` first; if Git on Windows reports "Filename too long" when
cloning, clone into a shorter folder such as `C:\src`.

GitHub repeats this check on Linux, macOS and Windows after every change and every Monday; the badge above shows the
result. If a newer numpy or pandas release ever makes a file differ, install the versions shown in the "Show the
versions" step of the last passing run (now numpy 2.5.4 and pandas 3.1.0 with Python 3.13).

## Sources

- Cases: Arizona Department of Health Services (ADHS), "MMWR by County" workbooks.
- Weather: Arizona Meteorological Network (AZMET), University of Arizona: Maricopa Agricultural Center (<https://azmet.arizona.edu/azmet/06.htm>), Tucson (<https://azmet.arizona.edu/azmet/01.htm>) and Coolidge (<https://azmet.arizona.edu/azmet/05.htm>).
- PM10 and AQI: U.S. Environmental Protection Agency (EPA), Outdoor Air Quality Data: <https://www.epa.gov/outdoor-air-quality-data>.

## License

The code is released under the MIT License (see [LICENSE](LICENSE)). The data remain under the terms of their sources (ADHS, AZMET and the U.S. EPA).
