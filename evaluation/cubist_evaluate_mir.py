import pandas as pd
from cubist import Cubist
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt

def load_data(test_data_path):
    # Load testing data from CSV files
    test_data = pd.read_csv(test_data_path)
    return test_data

def preprocess_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Apply log1p transformation to the target column
    y_log1p = np.log1p(y)

    return X, y_log1p


def evaluate_model(y_true, y_pred):
    # Evaluate the model using RMSE, LCCC, and R²
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    lccc = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r2_score(y_true, y_pred)
    return rmse, lccc, r2

def plot_evaluation(y_true, y_pred, rmse, lccc, r2, output_image_path, target_column):
    # Create a scatter plot of the logged predicted vs actual values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='blue', alpha=0.5, label='Predicted vs Actual')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', label='1:1 Line')
    plt.title(f'Logged Predicted vs Logged Actual for {target_column}')
    plt.xlabel('Logged Actual (log1p)')
    plt.ylabel('Logged Predicted (log1p)')
    plt.legend()
    
    # Annotate with RMSE, LCCC, and R²
    textstr = f'RMSE: {rmse:.4f}\nLCCC: {lccc:.4f}\nR²: {r2:.4f}'
    plt.gcf().text(0.03, 0.9, textstr, fontsize=13)
    # Save the plot
    plt.savefig(output_image_path)
    plt.close()

if __name__ == "__main__":
    # Load configuration from JSON file
    with open('./../data_processing/config.json', 'r') as f:
        config = json.load(f)

    # File paths for testing datasets
    spectra_type = "mir"  # Adjust as needed

    for target_column in config['targets']:
        # Loop over each target column in the configuration file
        print(f"Evaluating model for target: {target_column}")

        # File paths for testing datasets for the current target
        test_data_path = f'./../train_test_dataset/test_data_{spectra_type}_{target_column}.csv'

        # Load test data
        print(f"Loading test data for target: {target_column}")
        test_data = load_data(test_data_path)

        # Preprocess test data
        X_test, y_test = preprocess_data(test_data, target_column)

        # Load the trained model
        model_path = f'./../trained_models/cubist_model_{spectra_type}_{target_column}_logged.joblib'
        print(f"Loading model from: {model_path}")
        model = joblib.load(model_path)

        # Make predictions on the testing data
        y_pred = model.predict(X_test)

        # Evaluate the model
        rmse, lccc, r2 = evaluate_model(y_test, y_pred)
        print(f"RMSE: {rmse:.4f}, LCCC: {lccc:.4f}, R²: {r2:.4f}")

        # Output the image of predicted vs actual values
        output_image_path = f'./../evaluation_results/{spectra_type}_{target_column}_evaluation.png'
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        plot_evaluation(y_test, y_pred, rmse, lccc, r2, output_image_path, target_column)
        print(f"Evaluation image saved to: {output_image_path}")
