import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load and clean data from cleaned_filtered_data_V2.csv
input_file_path = r'C:/Users/merie/OneDrive/Bureau/WGU/D602/Task2/d602-deployment-task-2/ML_Project/cleaned_filtered_data_V2.csv'

data = pd.read_csv(input_file_path)

# Convert time columns to datetime
time_columns = ['SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME']

for col in time_columns:
    if col in data.columns:
        # Convert to datetime, coercing errors to NaT for invalid values
        data[col] = pd.to_datetime(data[col], format='%H:%M:%S', errors='coerce')
        
        # Check if the conversion was successful
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            # Convert time to seconds
            data[col] = data[col].dt.hour * 3600 + data[col].dt.minute * 60 + data[col].dt.second
        else:
            print(f"Column '{col}' could not be converted to datetime.")

# Extract the features and target variable
X = data[['SCHEDULED_DEPARTURE', 'DEPARTURE_DELAY']].values
y = data['ARRIVAL_DELAY'].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model parameters
alpha = 0.5
order = 2

# Transform the features using PolynomialFeatures
poly = PolynomialFeatures(degree=order)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train a Ridge regression model
model = Ridge(alpha=alpha)
model.fit(X_train_poly, y_train)

# Make predictions
y_pred = model.predict(X_test_poly)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
average_delay = y_test.mean()

# Start an MLFlow experiment
with mlflow.start_run(run_name="Final Model - Polynomial Regression") as run:
    # 1. Log the informational log files generated from the import_data and clean_data scripts
    # Log the cleaned dataset as an artifact
    mlflow.log_artifact(input_file_path)  
    
    # 2. Log the input parameters (alpha and order) to the final regression against the test data
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("order", order)
    
    # 3. Log the performance plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("True vs. Predicted Arrival Delays")
    plt.savefig("performance_plot.png")
    mlflow.log_artifact("performance_plot.png")
    
    # 4. Log the model performance metrics (mean squared error and the average delay in minutes)
    mlflow.log_metric("Mean Squared Error", mse)
    mlflow.log_metric("Average Delay", average_delay)

    print(f"Run ID: {run.info.run_id}")

mlflow.end_run()
