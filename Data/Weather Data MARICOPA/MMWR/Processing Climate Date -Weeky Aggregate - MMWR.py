import os
import pandas as pd
from datetime import datetime, timedelta


# Define the base year and format the years based on the last two digits in the filename
def get_file_year(filename):
    try:
        year_suffix = int(filename[2:4])  # Extracts the last two digits before "rd"
        if 87 <= year_suffix <= 99:
            year = 1900 + year_suffix  # Files from 1987 to 1999
        elif 0 <= year_suffix <= 24:
            year = 2000 + year_suffix  # Files from 2000 to 2024
        else:
            year = None  # Unknown year format
        print(f"File: {filename}, Year extracted: {year}")  # Debugging line
        return year
    except ValueError:
        print(f"Unable to determine year from filename: {filename}")
        return None  # Return None if year extraction fails


# Function to remove non-printable characters
def clean_line(line):
    import string
    return ''.join(filter(lambda x: x in string.printable, line))


# Define unified headers for final DataFrame
unified_headers = [
    "Year", "Day of Year", "Station Number", "Air Temp - Max", "Air Temp - Min", "Air Temp - Mean",
    "RH - Max", "RH - Min", "RH - Mean", "VPD - Mean", "Solar Rad. - Total", "Precipitation - Total",
    "4\" Soil Temp - Max", "4\" Soil Temp - Min", "4\" Soil Temp - Mean",
    "20\" Soil Temp - Max", "20\" Soil Temp - Min", "20\" Soil Temp - Mean",
    "Wind Speed - Mean", "Wind Vector Magnitude", "Wind Vector Direction",
    "Wind Direction Std Dev", "Max Wind Speed", "Heat Units", "Reference ET",
    "Reference ET (Penman-Monteith)", "Actual Vapor Pressure - Daily Mean", "Dewpoint - Daily Mean"
]

# Define data structure for 1987-2002 and 2003-present formats
headers_1987_2002 = [
    "Year", "Day of Year", "Station Number", "Air Temp - Max", "Air Temp - Min", "Air Temp - Mean",
    "RH - Max", "RH - Min", "RH - Mean", "VPD - Mean", "Solar Rad. - Total", "Precipitation - Total",
    "2\" Soil Temp - Max", "2\" Soil Temp - Min", "2\" Soil Temp - Mean",
    "4\" Soil Temp - Max", "4\" Soil Temp - Min", "4\" Soil Temp - Mean",
    "Wind Speed - Mean", "Wind Vector Magnitude", "Wind Vector Direction",
    "Wind Direction Std Dev", "Max Wind Speed", "Reference ET", "Heat Units"
]

headers_2003_present = [
    "Year", "Day of Year", "Station Number", "Air Temp - Max", "Air Temp - Min", "Air Temp - Mean",
    "RH - Max", "RH - Min", "RH - Mean", "VPD - Mean", "Solar Rad. - Total", "Precipitation - Total",
    "4\" Soil Temp - Max", "4\" Soil Temp - Min", "4\" Soil Temp - Mean",
    "20\" Soil Temp - Max", "20\" Soil Temp - Min", "20\" Soil Temp - Mean",
    "Wind Speed - Mean", "Wind Vector Magnitude", "Wind Vector Direction",
    "Wind Direction Std Dev", "Max Wind Speed", "Heat Units", "Reference ET",
    "Reference ET (Penman-Monteith)", "Actual Vapor Pressure - Daily Mean", "Dewpoint - Daily Mean"
]


# Function to calculate MMWR week and year

from datetime import datetime, timedelta

from datetime import datetime, timedelta


def calculate_mmwr_week(date):
    """
    Calculate the MMWR week and year for a given date based on the official CDC definition:
    MMWR Week 1 is the week that includes the first Thursday of the year. Each week starts on Sunday.
    """
    # Normalize to the start of the week (Sunday)
    # weekday(): Monday=0, Sunday=6, so (weekday()+1)%7 gives days since last Sunday.
    week_start = date - timedelta(days=(date.weekday() + 1) % 7)

    # Tentatively set mmwr_year to the calendar year of the given date
    mmwr_year = date.year

    # Step 1: Find the anchor Sunday (the Sunday of the week that includes January 4th of mmwr_year)
    january_4 = datetime(mmwr_year, 1, 4)
    anchor_sunday = january_4 - timedelta(days=(january_4.weekday() + 1) % 7)
    # Now, anchor_sunday marks the start of MMWR Week 1 for mmwr_year.

    # If the date is before the anchor Sunday, it actually belongs to the previous MMWR year.
    if week_start < anchor_sunday:
        mmwr_year -= 1
        # Recalculate the anchor_sunday for the previous year
        january_4_prev = datetime(mmwr_year, 1, 4)
        anchor_sunday = january_4_prev - timedelta(days=(january_4_prev.weekday() + 1) % 7)

    # Calculate how many days separate this week's Sunday from the anchor Sunday
    delta_days = (week_start - anchor_sunday).days
    mmwr_week = delta_days // 7 + 1

    # Handle the transition at the end of the MMWR year:
    # Find the last Sunday of the MMWR year by checking December 31 of mmwr_year.
    dec_31 = datetime(mmwr_year, 12, 31)
    last_sunday = dec_31 - timedelta(days=(dec_31.weekday() + 1) % 7)
    # If the last week of the mmwr_year has fewer than 4 days of that year, it's Week 1 of next year.
    # Check how many days from last_sunday to year end
    days_in_last_week = (dec_31 - last_sunday).days + 1  # inclusive count
    if days_in_last_week < 4 and week_start >= last_sunday:
        mmwr_week = 1
        mmwr_year += 1

    return mmwr_year, mmwr_week


# Add MMWR week to the DataFrame
def add_mmwr_weeks(df):
    df["MMWR_Year_Week"] = df["Date"].apply(calculate_mmwr_week)
    df["MMWR_Year"] = df["MMWR_Year_Week"].apply(lambda x: x[0])
    df["MMWR_Week"] = df["MMWR_Year_Week"].apply(lambda x: x[1])
    return df
# Updated processing function
def process_files(folder_path, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    combined_data = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_year = get_file_year(file_name)
            if file_year is None:
                print(f"Skipping file due to unknown year format: {file_name}")
                continue  # Skip files where the year couldn't be determined

            file_path = os.path.join(folder_path, file_name)
            print(f"Processing file: {file_name}")

            # Determine the format based on the year
            if 1987 <= file_year <= 2002:
                headers = headers_1987_2002
                delimiter_count = 25
            elif file_year >= 2003:
                headers = headers_2003_present
                delimiter_count = 28
            else:
                print(f"Skipping unknown file format for {file_name}")
                continue

            with open(file_path, 'r') as infile:
                for line in infile:
                    cleaned_line = clean_line(line)
                    data = cleaned_line.strip().split(',')
                    if len(data) == delimiter_count:
                        if file_year <= 2002:
                            if len(data[0]) == 2:
                                data[0] = '19' + data[0]

                        if file_year <= 2002:
                            data += [None, None, None]
                            data = data[:24] + [data[25], data[24]] + data[26:]

                        combined_data.append(data)
                    else:
                        print(f"Skipped malformed line in {file_name}: {repr(cleaned_line)}")

            print(f"Finished processing file: {file_name}")

    if combined_data:
        combined_df = pd.DataFrame(combined_data, columns=unified_headers)
        combined_df["Year"] = pd.to_numeric(combined_df["Year"], errors="coerce").astype("Int64")
        combined_df["Day of Year"] = pd.to_numeric(combined_df["Day of Year"], errors="coerce").astype("Int64")

        combined_df = combined_df.sort_values(by=["Year", "Day of Year"]).reset_index(drop=True)

        for col in combined_df.columns:
            if col not in ["Year", "Day of Year", "Station Number"]:
                combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

        combined_df["Year"] = pd.to_numeric(combined_df["Year"], errors="coerce").astype("Int64")
        combined_df["Day of Year"] = pd.to_numeric(combined_df["Day of Year"], errors="coerce").astype("Int64")

        combined_df = combined_df[combined_df["Year"].between(1900, 2100) & combined_df["Day of Year"].between(1, 366)]

        combined_df["Date"] = pd.to_datetime(
            combined_df["Year"].astype(str) + combined_df["Day of Year"].astype(str).str.zfill(3),
            format='%Y%j', errors='coerce'
        )
        combined_df.dropna(subset=["Date"], inplace=True)

        # Add MMWR week information
        combined_df = add_mmwr_weeks(combined_df)

        # Aggregate data by MMWR weeks
        # Exclude non-aggregatable columns from numeric_cols
        numeric_cols = combined_df.select_dtypes(include="number").columns.difference(
            ["MMWR_Year", "MMWR_Week"]).tolist()


        weekly_df = combined_df.groupby(["MMWR_Year", "MMWR_Week"])[numeric_cols].mean().reset_index()

        output_excel = os.path.join(output_directory, "combined_weather_data.xlsx")
        weekly_output_excel = os.path.join(output_directory, "weekly_aggregated_weather_data_mmwr.xlsx")

        combined_df.to_excel(output_excel, index=False)
        weekly_df.to_excel(weekly_output_excel, index=False)

        print(f"Combined data written to {output_excel}")
        print(f"Weekly aggregated data written to {weekly_output_excel}")
    else:
        print("No valid data to write.")



# Specify the folder containing the text files and the output directory
folder_path = "C:/Users/sarab/Desktop/desktop/Valley Fever/Weather Data MARICOPA(used)/data MARICOPA"
output_directory = "C:/Users/sarab/Desktop/desktop/Valley Fever/Weather Data MARICOPA (MMWR Weeks)"
process_files(folder_path, output_directory)
