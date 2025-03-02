# import pandas as pd
# import os
#
# # Directory containing the CSV files
# data_directory = r"Z:\\PM10 Data"
#
# # List of input CSV files
# file_names = [
#     "daily_81102_2006.csv", "daily_81102_2007.csv", "daily_81102_2008.csv", "daily_81102_2009.csv",
#     "daily_81102_2010.csv", "daily_81102_2011.csv", "daily_81102_2012.csv", "daily_81102_2013.csv",
#     "daily_81102_2014.csv", "daily_81102_2015.csv", "daily_81102_2016.csv", "daily_81102_2017.csv",
#     "daily_81102_2018.csv", "daily_81102_2019.csv", "daily_81102_2020.csv", "daily_81102_2021.csv",
#     "daily_81102_2022.csv", "daily_81102_2023.csv", "daily_81102_2024.csv"
# ]
#
# # Output CSV file name
# output_file = "filtered_data_arizona_maricopa.csv"
#
# # Initialize an empty DataFrame to store filtered data
# filtered_data = pd.DataFrame()
#
# # Process each file
# for file_name in file_names:
#     file_path = os.path.join(data_directory, file_name)  # Construct full file path
#     if os.path.exists(file_path):  # Check if the file exists
#         # Read the CSV file with the correct delimiter
#         try:
#             df = pd.read_csv(file_path, delimiter=",")  # Updated to use comma as delimiter
#             print(f"Processing {file_name}")
#             print(f"Columns: {df.columns.tolist()}")  # Debug column names
#
#             # Clean column names to remove leading/trailing quotes and spaces
#             df.columns = df.columns.str.strip().str.replace('"', '')
#
#             # Check if required columns exist
#             if 'State Name' in df.columns and 'County Name' in df.columns:
#                 # Filter rows where State Name is Arizona and County Name is Maricopa
#                 filtered_df = df[(df['State Name'] == 'Arizona') & (df['County Name'] == 'Maricopa')]
#                 filtered_data = pd.concat([filtered_data, filtered_df], ignore_index=True)
#             else:
#                 print(f"Missing required columns in {file_name}")
#         except Exception as e:
#             print(f"Error reading {file_name}: {e}")
#     else:
#         print(f"File not found: {file_path}")
#
# # Write the filtered data to a CSV file if there's any data
# if not filtered_data.empty:
#     output_file_path = os.path.join(data_directory, output_file)
#     filtered_data.to_csv(output_file_path, index=False)
#     print(f"Filtered data has been saved to {output_file_path}")
# else:
#     print("No data matched the criteria.")
#




#
# import pandas as pd
# import os
#
# # Directory containing the CSV files
# data_directory = r"Z:\\PM10 Data"
#
# # List of input CSV files
# file_names = [
#     "daily_81102_2006.csv", "daily_81102_2007.csv", "daily_81102_2008.csv", "daily_81102_2009.csv",
#     "daily_81102_2010.csv", "daily_81102_2011.csv", "daily_81102_2012.csv", "daily_81102_2013.csv",
#     "daily_81102_2014.csv", "daily_81102_2015.csv", "daily_81102_2016.csv", "daily_81102_2017.csv",
#     "daily_81102_2018.csv", "daily_81102_2019.csv", "daily_81102_2020.csv", "daily_81102_2021.csv",
#     "daily_81102_2022.csv", "daily_81102_2023.csv", "daily_81102_2024.csv"
# ]
#
# # Output CSV file name
# output_file = "Z:/PM10 Data/filtered_averaged_data_maricopa.csv"
#
# # Initialize an empty DataFrame to store filtered and aggregated data
# filtered_data = pd.DataFrame()
#
# # Process each file
# for file_name in file_names:
#     file_path = os.path.join(data_directory, file_name)  # Construct full file path
#     if os.path.exists(file_path):  # Check if the file exists
#         # Read the CSV file
#         try:
#             df = pd.read_csv(file_path, delimiter=",")  # Assuming comma as delimiter
#             print(f"Processing {file_name}")
#
#             # Clean column names to remove leading/trailing quotes and spaces
#             df.columns = df.columns.str.strip().str.replace('"', '')
#
#             # Check if required columns exist
#             if {'State Name', 'County Name', 'Method Name', 'Date Local', 'Arithmetic Mean', 'AQI'}.issubset(df.columns):
#                 # Filter for Arizona, Maricopa, and the specified Method Name
#                 filtered_df = df[
#                     (df['State Name'] == 'Arizona') &
#                     (df['County Name'] == 'Maricopa') &
#                     (df['Method Name'] == 'INSTRUMENTAL-R&P SA246B-INLET - TEOM-GRAVIMETRIC')
#                 ]
#                 # Append to the combined DataFrame
#                 filtered_data = pd.concat([filtered_data, filtered_df], ignore_index=True)
#             else:
#                 print(f"Missing required columns in {file_name}")
#         except Exception as e:
#             print(f"Error reading {file_name}: {e}")
#     else:
#         print(f"File not found: {file_path}")
#
# # If data is present, average by Date Local
# if not filtered_data.empty:
#     # Convert Date Local to datetime for proper grouping
#     filtered_data['Date Local'] = pd.to_datetime(filtered_data['Date Local'])
#
#     # Group by Date Local and calculate averages for Arithmetic Mean and AQI
#     averaged_data = (
#         filtered_data.groupby('Date Local', as_index=False)
#         .agg({'Arithmetic Mean': 'mean', 'AQI': 'mean'})
#     )
#
#     # Save the result to a CSV file
#     output_file_path = os.path.join(data_directory, output_file)
#     averaged_data.to_csv(output_file_path, index=False)
#     print(f"Averaged data has been saved to {output_file_path}")
# else:
#     print("No data matched the criteria.")



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
file_path = 'filtered_averaged_data_maricopa.csv'  # Replace with your file path
filtered_averaged_data = pd.read_csv(file_path)

# Convert the date column to datetime
filtered_averaged_data['Date'] = pd.to_datetime(filtered_averaged_data['Date Local'], format='%Y-%m-%d', errors='coerce')



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