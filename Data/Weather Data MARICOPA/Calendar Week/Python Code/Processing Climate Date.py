import os
import csv
import string

# Define the base year and format the years based on the last two digits in the filename
def get_file_year(filename):
    try:
        year_suffix = int(filename[2:4])  # Extracts the last two digits before "rd"
        if 0 <= year_suffix <= 24:
            return 2000 + year_suffix  # Files from 2000 to 2024
        else:
            return 1900 + year_suffix  # Files from 1987 to 1999
    except ValueError:
        print(f"Unable to determine year from filename: {filename}")
        return None  # Return None if year extraction fails

# Function to remove non-printable characters
def clean_line(line):
    return ''.join(filter(lambda x: x in string.printable, line))

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
    "Wind Direction Std Dev", "Max Wind Speed", "Heat Units",
    "Reference ET (Original AZMET)", "Reference ET (Penman-Monteith)",
    "Actual Vapor Pressure - Daily Mean", "Dewpoint - Daily Mean"
]

# Function to process the files
def process_files(folder_path, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_year = get_file_year(file_name)
            if file_year is None:
                continue  # Skip files where the year couldn't be determined

            file_path = os.path.join(folder_path, file_name)

            if 1987 <= file_year <= 2002:  # 1987-2002 format
                headers = headers_1987_2002
                delimiter_count = 25  # 25 data points for 1987-2002
            elif file_year >= 2003:  # 2003-present format
                headers = headers_2003_present
                delimiter_count = 28  # 28 data points for 2003-present
            else:
                print(f"Skipping unknown file format for {file_name}")
                continue

            output_csv = os.path.join(output_directory, file_name.replace(".txt", ".csv"))

            with open(file_path, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(headers)

                for line in infile:
                    cleaned_line = clean_line(line)
                    data = cleaned_line.strip().split(',')
                    if len(data) == delimiter_count:
                        writer.writerow(data)
                    else:
                        print(f"Skipped malformed line in {file_name}: {repr(cleaned_line)}")

            print(f"Converted {file_name} to {output_csv}")




# Specify the folder containing the text files and the output directory
folder_path = "C:/Users/sarab/Desktop/desktop/Valley Fever/Weather Data MARICOPA/data MARICOPA"
output_directory = "C:/Users/sarab/Desktop/desktop/Valley Fever/Weather Data MARICOPA"
process_files(folder_path, output_directory)

