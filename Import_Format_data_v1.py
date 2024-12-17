import pandas as pd

# Load the data
input_file_path = "C:/Users/merie/OneDrive/Bureau/WGU/D602/Task2/d602-deployment-task-2/data/Airline_Delay.csv"
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

# Handle time-related columns 
df['SCHEDULED_DEPARTURE'] = pd.to_datetime(df['SCHEDULED_DEPARTURE'], format='%H%M', errors='coerce')
df['DEPARTURE_TIME'] = pd.to_datetime(df['DEPARTURE_TIME'], format='%H%M', errors='coerce')
df['SCHEDULED_ARRIVAL'] = pd.to_datetime(df['SCHEDULED_ARRIVAL'], format='%H%M', errors='coerce')
df['ARRIVAL_TIME'] = pd.to_datetime(df['ARRIVAL_TIME'], format='%H%M', errors='coerce')

# Save the formatted data to a new CSV file
output_file_path = "C:/Users/merie/OneDrive/Bureau/WGU/D602/Task2/d602-deployment-task-2/data/formatted_data.csv"
df.to_csv(output_file_path, index=False)

print("Formatted data saved to:", output_file_path)
