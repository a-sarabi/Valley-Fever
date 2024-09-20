# import pandas as pd
#
# # Path to the Excel file
# file_path = 'Calendar Week - ASU.xlsx'
#
# # Name of the introduction sheet to exclude
# introduction_sheet_name = 'Introduction'
#
# # Columns to select
# columns_to_select = ['Unnamed: 0', 'MARICOPA', 'PIMA', 'PINAL']
#
# # Initialize an empty DataFrame to hold combined data
# combined_df = pd.DataFrame()
#
# # Loop through each sheet in the Excel file
# with pd.ExcelFile(file_path) as xls:
#     for sheet_name in xls.sheet_names:
#         if sheet_name != introduction_sheet_name:
#             # Read each sheet into a DataFrame, skip the first two rows and use the third row as header
#             df = pd.read_excel(xls, sheet_name=sheet_name, skiprows=2)
#             # Select only the specified columns
#             df = df[columns_to_select]
#             # Drop the last row (Total row)
#             df = df.iloc[:-1]
#             # Add a column to identify the year
#             df['Year'] = sheet_name
#
#             # Create a new column "Year_Week" by combining "Unnamed: 0" and "Year"
#             df['Year_Week'] = df['Year'] + "_" + df['Unnamed: 0'].astype(str)
#             # Drop the original "Unnamed: 0" and "Year" columns
#             df = df.drop(columns=['Unnamed: 0', 'Year'])
#             # Reorder columns to place "Year_Week" as the first column
#             df = df[['Year_Week'] + [col for col in df.columns if col != 'Year_Week']]
#
#             # Append the DataFrame to the combined DataFrame
#             combined_df = pd.concat([combined_df, df], ignore_index=True)
#
# # Save the combined DataFrame to a new Excel file
# combined_df.to_excel('combined_data.xlsx', index=False)
#
# print("Data combined successfully!")


###############################################
# Add your NOAA CDO token here.
token = 'FdZNSPwqKckygcLANvsyadvRvaFdHKGd'

# Import necessary libraries
import requests
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import time  # For handling rate limits

print('Ready to go!')

# Set the initial and final years you want data from.
yearI = 1990  # Initial year.
yearF = 2023  # Final year.

# Read GHCND stations file and filter by Arizona state and specified counties.
station_file = 'C:/Users/sarab/Desktop/desktop/Valley Fever/Data/ghcnd-stations.txt'

stations_data = []
with open(station_file, 'r') as file:
    for line in file:
        parts = line.split()
        if len(parts) >= 6:  # Ensure minimum number of fields are present
            station_id = parts[0]
            latitude = float(parts[1])
            longitude = float(parts[2])
            elevation = float(parts[3])
            state = parts[4] if len(parts[4]) == 2 else None
            name = ' '.join(parts[5:])
            stations_data.append([station_id, latitude, longitude, elevation, state, name])

stations_df = pd.DataFrame(stations_data, columns=['ID', 'Latitude', 'Longitude', 'Elevation', 'State', 'Name'])

county_keywords_dicts = {
    "Maricopa": [
                'MARICOPA', 'PHOENIX',
                 'TEMPE', 'MESA',
                 'SCOTTSDALE', 'GLENDALE', 'CHANDLER',
                 'GILBERT', "PEORIA", "SURPRISE",
                 "AVONDALE", "GOODYEAR", "BUCKEYE",
                 "EL MIRAGE", "FOUNTAIN HILLS",
                 "PARADISE VALLEY", "TOLLESON"
                 ],

    "Pinal": ["CASA GRANDE", "FLORENCE",
              "APACHE JUNCTION", "COOLIDGE",
              "ELOY", 'MARICOPA'
              ],

    "Pima": ['TUCSON', "ORO VALLEY",

             "SAHUARITA", 'MARANA', 'SOUTH TUCSON'
             ]
}


def contains_county(name_parts, counties):
    """Check if any county name is present in the station name."""
    return any(county in name_parts for county in counties)


def fetch_data_for_station(station_id):
    data_list = []

    for year in range(yearI, yearF + 1):
        year_str = str(year)
        print(f'Working on year {year_str} for station {station_id}')

        offset = 1  # Initialize offset before the pagination loop
        while True:
            # Build the URL with offset and limit
            url = f'https://www.ncdc.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND'
            url += f'&datatypeid=TMAX&datatypeid=TMIN&datatypeid=PRCP'
            url += f'&stationid=GHCND:{station_id}&startdate={year_str}-01-01&enddate={year_str}-12-31'
            url += f'&limit=1000&offset={offset}'

            response = requests.get(url, headers={'token': token})

            if response.status_code == 200:
                data = response.json()

                if "results" in data:
                    results_len = len(data['results'])
                    print(f"Retrieved {results_len} records for {year}, offset {offset}")

                    # Process data...
                    for item in data['results']:
                        data_list.append({
                            'date': item['date'],
                            'datatype': item['datatype'],
                            'value': item['value'],
                            'station': station_id
                        })

                    # Break if we've retrieved the last page
                    if results_len < 1000:
                        break

                    # Increment offset by limit
                    offset += 1000

                else:
                    break  # No results found, exit loop

            elif response.status_code == 429:
                print(f"Rate limit exceeded. Waiting for 60 seconds...")
                time.sleep(60)
                continue

            elif response.status_code == 503:
                print(f"Service unavailable for {year}. Retrying...")
                time.sleep(5)
                continue

            else:
                print(f"Failed to retrieve data for {year}: {response.status_code}")
                break

    return data_list



# Create a directory to store raw data files if it doesn't exist
raw_data_dir = 'raw_data'
if not os.path.exists(raw_data_dir):
    os.makedirs(raw_data_dir)

# Dictionary to store weekly county averages for all counties
county_weekly_averages = {}

for county_name, cities_identifiers in county_keywords_dicts.items():
    stations_df_filtered = stations_df[(stations_df['State'] == 'AZ') ]#| (stations_df['State'].isna())]

    stations_df_filtered['County'] = stations_df_filtered['Name'].apply(
        lambda x: contains_county(x.upper().split(), cities_identifiers))

    stations_of_interest = stations_df_filtered[stations_df_filtered['County']]['ID'].tolist()

    print(f"\nProcessing County: {county_name}")
    print(f"Found {len(stations_of_interest)} stations")

    #all_station_data = []

    for station_id in stations_of_interest:
        station_data = fetch_data_for_station(station_id)
        if not station_data:
            print(f"No data retrieved for station {station_id}.")
            continue

        # Convert to DataFrame
        df_station_data = pd.DataFrame(station_data)

        # Save raw data to Excel file
        raw_data_filename = f"{raw_data_dir}/{county_name}_{station_id}_raw_data.xlsx"
        df_station_data.to_excel(raw_data_filename, index=False)
        print(f"Raw data for station {station_id} saved to {raw_data_filename}")

        #all_station_data.append(df_station_data)
#
#     if not all_station_data:
#         print(f"No data retrieved for any stations in {county_name} County.")
#         continue
#
#     # Combine data from all stations in the county
#     df_all_data = pd.concat(all_station_data, ignore_index=True)
#
#     # Convert 'date' to datetime
#     df_all_data['date'] = pd.to_datetime(df_all_data['date'])
#
#     # Pivot the data to have columns for each datatype
#     df_pivot = df_all_data.pivot_table(index=['station', 'date'], columns='datatype', values='value').reset_index()
#
#     # # Convert units
#     # df_pivot['AWND'] = df_pivot['AWND'] / 10.0  # Convert to m/s
#     # df_pivot['TAVG'] = df_pivot['TAVG'] / 10.0  # Convert to degrees Celsius
#     # df_pivot['PRCP'] = df_pivot['PRCP'] / 10.0  # Convert to mm
#
#     # Save the combined data to Excel file
#     combined_data_filename = f"{county_name}_combined_raw_data.xlsx"
#     df_pivot.to_excel(combined_data_filename, index=False)
#     print(f"Combined raw data for {county_name} County saved to {combined_data_filename}")
#
#     # Compute weekly aggregates for each station
#     df_pivot.set_index('date', inplace=True)
#     weekly_station_avg = df_pivot.groupby(['station', pd.Grouper(freq='W')]).mean().reset_index()
#
#     # Save the weekly aggregates to Excel file
#     weekly_station_avg_filename = f"{county_name}_weekly_station_averages.xlsx"
#     weekly_station_avg.to_excel(weekly_station_avg_filename, index=False)
#     print(f"Weekly station averages for {county_name} County saved to {weekly_station_avg_filename}")
#
#     # Compute weekly aggregates across all stations
#     weekly_county_avg = df_pivot.groupby(pd.Grouper(freq='W')).mean().reset_index()
#
#     # Store the county weekly averages in the dictionary
#     county_weekly_averages[county_name] = weekly_county_avg
#
#     # Save the weekly county averages to Excel file
#     weekly_county_avg_filename = f"{county_name}_weekly_county_averages.xlsx"
#     weekly_county_avg.to_excel(weekly_county_avg_filename, index=False)
#     print(f"Weekly county averages for {county_name} County saved to {weekly_county_avg_filename}")
#
#     # Plotting (optional)
#     plt.figure(figsize=(12, 6))
#     plt.plot(weekly_county_avg['date'], weekly_county_avg['TAVG'], label='Average Temperature (°C)')
#     plt.plot(weekly_county_avg['date'], weekly_county_avg['AWND'], label='Average Wind Speed (m/s)')
#     plt.plot(weekly_county_avg['date'], weekly_county_avg['PRCP'], label='Precipitation (mm)')
#     plt.xlabel('Date')
#     plt.ylabel('Values')
#     plt.title(f'Weekly Weather Averages ({county_name} County)')
#     plt.legend()
#     plt.show()
#
# # After processing all counties, save the county averages to a separate Excel file
# output_filename = "all_counties_weekly_averages.xlsx"
# with pd.ExcelWriter(output_filename) as writer:
#     for county_name, df_county_avg in county_weekly_averages.items():
#         df_county_avg.to_excel(writer, sheet_name=county_name, index=False)
# print(f"All counties' weekly averages saved to {output_filename}")

