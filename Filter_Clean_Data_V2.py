import pandas as pd
import subprocess

# Load the data
input_file_path = "C:/Users/merie/OneDrive/Bureau/WGU/D602/Task2/d602-deployment-task-2/ML_Project/Airline_Delay.csv"
df = pd.read_csv(input_file_path)

# Check the columns in the dataset
print("Columns in the dataset:", df.columns)

# Rename columns to match the model
df.rename(columns={
    'ORIGIN': 'ORG_AIRPORT',
    'DEST': 'DEST_AIRPORT',
    'CRS_DEP_TIME': 'SCHEDULED_DEPARTURE',
    'DEP_TIME': 'DEPARTURE_TIME',
    'DEP_DELAY': 'DEPARTURE_DELAY',
    'CRS_ARR_TIME': 'SCHEDULED_ARRIVAL',
    'ARR_TIME': 'ARRIVAL_TIME',
    'ARR_DELAY': 'ARRIVAL_DELAY',
    'DAY_OF_MONTH': 'DAY'
}, inplace=True)

# Ensure the correct format for date columns
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])

# Handle time-related columns and strip out the date, keeping only the time
df['SCHEDULED_DEPARTURE'] = pd.to_datetime(df['SCHEDULED_DEPARTURE'], format='%H%M', errors='coerce').dt.time
df['DEPARTURE_TIME'] = pd.to_datetime(df['DEPARTURE_TIME'], format='%H%M', errors='coerce').dt.time
df['SCHEDULED_ARRIVAL'] = pd.to_datetime(df['SCHEDULED_ARRIVAL'], format='%H%M', errors='coerce').dt.time
df['ARRIVAL_TIME'] = pd.to_datetime(df['ARRIVAL_TIME'], format='%H%M', errors='coerce').dt.time

# 1. Filter the data for a specific departure airport (JFK)
chosen_airport = 'JFK'
df_filtered = df[df['ORG_AIRPORT'] == chosen_airport]

# 2. Remove duplicate rows
df_filtered = df_filtered.drop_duplicates()

# 3. Handle missing values by dropping rows with missing data
df_filtered = df_filtered.dropna()

# Save the cleaned and filtered data to a new CSV file
output_file_path = "C:/Users/merie/OneDrive/Bureau/WGU/D602/Task2/d602-deployment-task-2/ML_Project/cleaned_filtered_data_V2.csv"
df_filtered.to_csv(output_file_path, index=False)

print("Cleaned and filtered data verion 2 saved to:", output_file_path)


