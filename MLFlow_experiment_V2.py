import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load and clean data from cleaned_filtered_data_V2.csv
input_file_path = r'C:/Users/merie/OneDrive/Bureau/WGU/D602/Task2/d602-deployment-task-2/ML_Project/cleaned_filtered_data_V2.csv'
data = pd.read_csv(input_file_path)

# Convert time columns to datetime and transform to seconds
time_columns = ['SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME']
for col in time_columns:
    if col in data.columns:
        data[col] = pd.to_datetime(data[col], format='%H:%M:%S', errors='coerce')
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            data[col] = data[col].dt.hour * 3600 + data[col].dt.minute * 60 + data[col].dt.second

# Extract features and target variable
X = data[['SCHEDULED_DEPARTURE', 'DEPARTURE_DELAY']].values
y = data['ARRIVAL_DELAY'].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grid for tuning
param_grid = {
    'alpha': [0.1, 0.5, 1, 10],
    'order': [1, 2, 3]
}

# Grid search with cross-validation
best_model = None
best_mse = float('inf')

for alpha in param_grid['alpha']:
    for order in param_grid['order']:
        poly = PolynomialFeatures(degree=order)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        model = Ridge(alpha=alpha)
        model.fit(X_train_poly, y_train)

        y_pred = model.predict(X_test_poly)
        mse = mean_squared_error(y_test, y_pred)

        # Log the best model
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_alpha = alpha
            best_order = order

# Start MLFlow experiment for version 2
with mlflow.start_run(run_name="Polynomial Regression - Version 2") as run:
    # Log the dataset as artifact
    mlflow.log_artifact(input_file_path)
    
    # Log the best hyperparameters
    mlflow.log_param("Best Alpha", best_alpha)
    mlflow.log_param("Best Order", best_order)
    
    # Log the final model
    mlflow.sklearn.log_model(best_model, "best_model")
    
    # Log performance metrics
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title(f"True vs Predicted Arrival Delays (Alpha={best_alpha}, Order={best_order})")
    plt.savefig("performance_plot_v2.png")
    mlflow.log_artifact("performance_plot_v2.png")
    
    mlflow.log_metric("Best Mean Squared Error", best_mse)
    mlflow.log_metric("Average Delay", y_test.mean())
    
    print(f"Run ID: {run.info.run_id}")

mlflow.end_run()


# JSON to use for task 3
import mlflow
import json
import pandas as pd

# Set the tracking URI if necessary (default is the local file system)
mlflow.set_tracking_uri('file:///C:/Users/merie/OneDrive/Bureau/WGU/D602/Task2/d602-deployment-task-2/ML_Project/mlruns')

# Replace with your actual run_id from the experiment
run_id = "11d7f23178544bd9abb0cd2c45628810" 

# Get the details of the mlflow run
run_details = {
    "run_id": run.info.run_id,
    "start_time": run.info.start_time,
    "end_time": run.info.end_time,
    "params": run.data.params,
    "metrics": run.data.metrics
}

# Load the cleaned filtered data to extract airport information
input_file_path = r"C:/Users/merie/OneDrive/Bureau/WGU/D602/Task2/d602-deployment-task-2/ML_Project/cleaned_filtered_data_V2.csv"
df = pd.read_csv(input_file_path)

# Extract unique airport codes from the data (both origin and destination airports)
airport_codes = pd.concat([df['ORG_AIRPORT'], df['DEST_AIRPORT']]).unique()

# Create a dictionary to store airport details
airport_data = {}


for airport in airport_codes:
    airport_data[airport] = {
        "name": airport,  
        "location": "Unknown"  
    }

# Add airport data to run details
run_details["airport_data"] = airport_data

# Save the combined details to a JSON file
output_file_path = "C:/Users/merie/OneDrive/Bureau/WGU/D602/Task2/d602-deployment-task-2/ML_Project/airport_encodings.json"
with open(output_file_path, 'w') as f:
    json.dump(run_details, f, indent=4)

print(f"Run details and airport data saved to {output_file_path}")
