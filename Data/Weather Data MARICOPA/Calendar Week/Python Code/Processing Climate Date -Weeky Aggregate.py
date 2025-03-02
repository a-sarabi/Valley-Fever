import os
import pandas as pd


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


# Function to process the files and combine data into one DataFrame
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
            print(f"Processing file: {file_name}")  # Debugging line

            # Determine the format based on the year
            if 1987 <= file_year <= 2002:  # 1987-2002 format
                headers = headers_1987_2002
                delimiter_count = 25  # 25 data points for 1987-2002
            elif file_year >= 2003:  # 2003-present format
                headers = headers_2003_present
                delimiter_count = 28  # 28 data points for 2003-present
            else:
                print(f"Skipping unknown file format for {file_name}")
                continue

            with open(file_path, 'r') as infile:
                for line in infile:
                    cleaned_line = clean_line(line)
                    data = cleaned_line.strip().split(',')
                    # Check if the line matches the expected delimiter count for this file format
                    if len(data) == delimiter_count:
                        # Adjust two-digit year to four digits for data before the year 2000
                        if file_year <= 2002:
                            # Convert 2-digit year to 4-digit if the first column is 2 digits
                            if len(data[0]) == 2:
                                data[0] = '19' + data[0]  # Convert '98' to '1998', for example

                        # Continue adjusting data to match unified headers
                        if file_year <= 2002:  # Fill in missing columns for old format
                            data += [None, None, None]  # Add placeholders for new columns
                            # Rearrange data to match unified headers
                            data = data[:24] + [data[25], data[24]] + data[26:]

                        combined_data.append(data)
                    else:
                        print(f"Skipped malformed line in {file_name}: {repr(cleaned_line)}")

            print(f"Finished processing file: {file_name}")

    # Create DataFrame and write to Excel
    if combined_data:
        combined_df = pd.DataFrame(combined_data, columns=unified_headers)
        # Ensure Year and Day of Year are numeric for sorting
        combined_df["Year"] = pd.to_numeric(combined_df["Year"], errors="coerce").astype("Int64")
        combined_df["Day of Year"] = pd.to_numeric(combined_df["Day of Year"], errors="coerce").astype("Int64")

        # Sort by Year and Day of Year
        combined_df = combined_df.sort_values(by=["Year", "Day of Year"]).reset_index(drop=True)

        # Convert all numeric columns to numeric data types
        for col in combined_df.columns:
            if col not in ["Year", "Day of Year", "Station Number", "Calendar Week"]:
                combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

        # Convert Year and Day of Year to integers to ensure consistent data types
        combined_df["Year"] = pd.to_numeric(combined_df["Year"], errors="coerce").astype("Int64")
        combined_df["Day of Year"] = pd.to_numeric(combined_df["Day of Year"], errors="coerce").astype("Int64")

        # Filter out invalid Year and Day of Year values
        combined_df = combined_df[combined_df["Year"].between(1900, 2100) & combined_df["Day of Year"].between(1, 366)]

        # Add Date and Week columns
        combined_df["Date"] = pd.to_datetime(
            combined_df["Year"].astype(str) + combined_df["Day of Year"].astype(str).str.zfill(3),
            format='%Y%j', errors='coerce'
        )
        combined_df.dropna(subset=["Date"], inplace=True)  # Drop rows with invalid dates

        # Set Calendar Week starting from 0, with each week beginning on Sunday, in year_week format
        combined_df["Week"] = combined_df["Date"].dt.strftime("%U")  # Week numbers start from 0 on Sundays
        combined_df["Calendar Week"] = combined_df["Year"].astype(str) + "_" + combined_df["Week"]

        # Select numeric columns for aggregation
        numeric_cols = combined_df.select_dtypes(include="number").columns.tolist()
        weekly_df = combined_df.groupby("Calendar Week")[numeric_cols].mean().reset_index()

        # Save combined and weekly-aggregated data to Excel
        output_excel = os.path.join(output_directory, "combined_weather_data.xlsx")
        weekly_output_excel = os.path.join(output_directory, "weekly_aggregated_weather_data.xlsx")

        combined_df.to_excel(output_excel, index=False)
        weekly_df.to_excel(weekly_output_excel, index=False)

        print(f"Combined data written to {output_excel}")
        print(f"Weekly aggregated data written to {weekly_output_excel}")
    else:
        print("No valid data to write.")



# Specify the folder containing the text files and the output directory
folder_path = "C:/Users/sarab/Desktop/desktop/Valley Fever/Weather Data MARICOPA/data MARICOPA"
output_directory = "C:/Users/sarab/Desktop/desktop/Valley Fever/Weather Data MARICOPA"
process_files(folder_path, output_directory)
