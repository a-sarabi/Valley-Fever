import pandas as pd
import os
import glob

# Directory containing the CSV files
data_directory = r"D:/Shared/desktop 3/Valley Fever/PM10 Maricopa"

# Method code(s) to filter - set to None to include all methods
# Common method codes:
# 208: TEOM (Tapered Element Oscillating Microbalance)
# 118: Gravimetric/Manual Reference Method
# Set to a list for multiple methods: [208, 118]
METHOD_CODE_FILTER = 79  # Change to None to include all methods

# Get all Maricopa CSV files using pattern matching
# This will find files like "Maricopa 2006.csv", "Maricopa 2007.csv", etc.
file_pattern = os.path.join(data_directory, "Maricopa *.csv")
file_paths = glob.glob(file_pattern)

# Output CSV file name
output_file = os.path.join(data_directory, "filtered_averaged_data_maricopa_new.csv")

# Initialize an empty DataFrame to store filtered and aggregated data
filtered_data = pd.DataFrame()

# Process each file
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    if os.path.exists(file_path):
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            print(f"Processing {file_name}")

            # Clean column names to remove leading/trailing spaces
            df.columns = df.columns.str.strip()

            # Check if required columns exist
            required_columns = {'Date', 'Daily Mean PM10 Concentration', 'Daily AQI Value',
                                'State', 'County'}

            if required_columns.issubset(df.columns):
                # First, let's see what methods are in the data
                if 'Method Code' in df.columns:
                    unique_methods = df['Method Code'].unique()
                    print(f"  Found Method Codes: {unique_methods}")

                # Filter for Arizona and Maricopa (adjust if needed)
                # Based on your sample data, it looks like the data might already be filtered for Maricopa
                filtered_df = df[
                    (df['State'] == 'Arizona') &
                    (df['County'] == 'Maricopa')
                    ]

                # Filter by specific method code to ensure consistency
                # Method 208 appears to be TEOM-based from your sample data
                if 'Method Code' in df.columns and METHOD_CODE_FILTER is not None:
                    if isinstance(METHOD_CODE_FILTER, list):
                        filtered_df = filtered_df[filtered_df['Method Code'].isin(METHOD_CODE_FILTER)]
                        print(f"  Filtered to Method Codes {METHOD_CODE_FILTER}: {len(filtered_df)} records")
                    else:
                        filtered_df = filtered_df[filtered_df['Method Code'] == METHOD_CODE_FILTER]
                        print(f"  Filtered to Method Code {METHOD_CODE_FILTER}: {len(filtered_df)} records")

                # Append to the combined DataFrame
                filtered_data = pd.concat([filtered_data, filtered_df], ignore_index=True)
            else:
                print(f"Missing required columns in {file_name}")
                print(f"Available columns: {list(df.columns)}")
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
    else:
        print(f"File not found: {file_path}")

# If data is present, average by Date
if not filtered_data.empty:
    # Convert Date to datetime for proper grouping
    filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])

    # Extract year for additional grouping if needed
    filtered_data['Year'] = filtered_data['Date'].dt.year

    # Group by Date and calculate averages for PM10 and AQI
    # This will average across all sites for each date
    averaged_data = (
        filtered_data.groupby('Date', as_index=False)
        .agg({
            'Daily Mean PM10 Concentration': 'mean',
            'Daily AQI Value': 'mean'
        })
    )

    # Round the averaged values to reasonable precision
    averaged_data['Daily Mean PM10 Concentration'] = averaged_data['Daily Mean PM10 Concentration'].round(2)
    averaged_data['Daily AQI Value'] = averaged_data['Daily AQI Value'].round(1)

    # Sort by date
    averaged_data = averaged_data.sort_values('Date')

    # Save the result to a CSV file
    averaged_data.to_csv(output_file, index=False)
    print(f"\nAveraged data has been saved to {output_file}")
    print(f"Total days processed: {len(averaged_data)}")

    # Create a method summary if Method Code column exists
    if 'Method Code' in filtered_data.columns:
        method_summary = filtered_data.groupby('Method Code').agg({
            'Daily Mean PM10 Concentration': ['count', 'mean', 'std'],
            'Daily AQI Value': 'mean'
        })
        print("\nMethod Code Summary:")
        print(method_summary)
        if METHOD_CODE_FILTER is not None:
            print(f"\nNote: Only Method Code(s) {METHOD_CODE_FILTER} included in the averaged output.")
        else:
            print("\nNote: All method codes were included in the averaged output.")

#
#########################################################################################
import pandas as pd
from datetime import datetime, timedelta

# Define the MMWR week calculation function
def calculate_mmwr_week(date):
    """
    Calculate the MMWR week and year for a given date based on the CDC's definition.
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

# Load the data
file_path = 'filtered_averaged_data_maricopa_new.csv'  # Replace with your file path
filtered_averaged_data = pd.read_csv(file_path)

# Convert the date column to datetime
filtered_averaged_data['Date'] = pd.to_datetime(filtered_averaged_data['Date'], format='%Y-%m-%d', errors='coerce')



# Add MMWR year and week columns
filtered_averaged_data['MMWR_Year_Week'] = filtered_averaged_data['Date'].apply(calculate_mmwr_week)
filtered_averaged_data['MMWR_Year'] = filtered_averaged_data['MMWR_Year_Week'].apply(lambda x: x[0])
filtered_averaged_data['MMWR_Week'] = filtered_averaged_data['MMWR_Year_Week'].apply(lambda x: x[1])




# Select numeric columns excluding 'MMWR_Year' and 'MMWR_Week'
numeric_cols = filtered_averaged_data.select_dtypes(include='number').columns.difference(['MMWR_Year', 'MMWR_Week'])


# Aggregate data by MMWR week/year
aggregated_data = filtered_averaged_data.groupby(["MMWR_Year", "MMWR_Week"])[numeric_cols].mean().reset_index()



# Save the aggregated data to a CSV file
output_csv_path = 'aggregated_mmwr_week_data.csv'  # Replace with your desired output path
aggregated_data.to_csv(output_csv_path, index=False)

print(f"Aggregated data saved to {output_csv_path}")