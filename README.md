# Valley Fever, Weather and PM10 Data for Three Arizona Counties

Weekly Valley fever (coccidioidomycosis) case counts for Maricopa, Pima and Pinal counties, Arizona, with weekly weather and PM10 (dust) data. The data cover MMWR weeks 2006_1 to 2026_39 (1 January 2006 to 3 October 2026).

## Contents

- `Data/Processed Data/`: the final weekly files, one per county.
- `Data/Maricopa/`, `Data/Pima (Tucson)/` and `Data/Pinal (Coolidge)/`: the original daily weather files (AZMET) and PM10 files (EPA) for each county.
- `Data/Valley Fever Data/`: the weekly case counts for each county (ADHS).
- `Data/Preprocessing/`: the program that builds the processed files from the original data.

[Data/README.md](Data/README.md) describes the sources, columns, units and processing.

## Using the data

Each processed file has one row per MMWR week, with the county's weekly number of cases, 25 weather variables, PM10 and the air quality index (AQI). For example, in Python:

```python
import pandas as pd

pima = pd.read_csv("Data/Processed Data/Clean Processed Data2_Pima.csv")
```

To rebuild the processed files from the original data (Python 3.9 or newer, with numpy and pandas 1.5 or newer):

```
cd Data/Preprocessing
python preprocess_counties.py
```

The program writes the files to `Data/Preprocessing/output/`; they are identical to those in `Data/Processed Data/`.

## Sources

- Cases: Arizona Department of Health Services (ADHS), "MMWR by County" workbooks.
- Weather: Arizona Meteorological Network (AZMET), University of Arizona: Maricopa Agricultural Center (<https://azmet.arizona.edu/azmet/06.htm>), Tucson (<https://azmet.arizona.edu/azmet/01.htm>) and Coolidge (<https://azmet.arizona.edu/azmet/05.htm>).
- PM10 and AQI: U.S. Environmental Protection Agency (EPA), Outdoor Air Quality Data: <https://www.epa.gov/outdoor-air-quality-data>.

## License

The code is released under the MIT License (see [LICENSE](LICENSE)). The data remain under the terms of their sources (ADHS, AZMET and the U.S. EPA).
