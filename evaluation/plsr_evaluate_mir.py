import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt
import csv

def load_data(merged_data, train_test_path, id_column, target_column, number_of_data):
    with open(train_test_path, 'r') as json_file:
        all_splits = json.load(json_file)

    key = f'{target_column}_{number_of_data}'

    if f'test_{key}' not in all_splits:
        return []

    test_ids = all_splits[f'test_{key}']

    print(f"Test ids : {len(test_ids)}")

    test_data = merged_data[merged_data[id_column].isin(test_ids)]

    print(f"Test data shape: {test_data.shape}")

    return test_data

def preprocess_data(data, target_column):
    spectral_columns = [col for col in data.columns if 'scan_mir' in col or 'scan_visnir' in col or 'scan_nir' in col]

    X = data[spectral_columns]
    y = data[target_column]

    if target_column in log_targets:
        y = np.log1p(y)

    return X, y

def evaluate_model(y_true, y_pred):
    # Evaluate the model using RMSE, LCCC, and R²
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    lccc = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r2_score(y_true, y_pred)
    return rmse, lccc, r2

def plot_evaluation(y_true, y_pred, rmse, lccc, r2, output_image_path, target_column):
    prefix = ''
    if target_column in log_targets:
        prefix = '(log1p)'

    # Create a scatter plot of the logged predicted vs actual values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='blue', alpha=0.5, label='Predicted vs Actual')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', label='1:1 Line')
    plt.title(f'{prefix} Predicted vs Actual for {target_column}')
    plt.xlabel(prefix+' Actual')
    plt.ylabel(prefix+' Predicted')
    plt.legend()
    
    # Annotate with RMSE, LCCC, and R²
    textstr = f'RMSE: {rmse:.4f}\nLCCC: {lccc:.4f}\nR²: {r2:.4f}'
    plt.gcf().text(0.03, 0.9, textstr, fontsize=13)
    # Save the plot
    plt.savefig(output_image_path)
    plt.close()

def log_evaluation_results(result, csv_file_path):
    # Check if the CSV file already exists
    file_exists = os.path.isfile(csv_file_path)

    # Open the CSV file in append mode
    with open(csv_file_path, 'a', newline='') as csvfile:
        fieldnames = ['model_used', 'parameters', 'no_of_samples', 'train_test_split_ratio', 'target', 'spectra_type',
                      'rmse', 'lccc', 'r_squared', 'logged', 'train_data_key', 'test_data_key']
        
        # Create a CSV writer object
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if the file is newly created
        if not file_exists:
            writer.writeheader()

        # Write the evaluation result to the CSV file
        writer.writerow(result)

    print(f"Evaluation result logged to: {csv_file_path}")

if __name__ == "__main__":
    folder_path = './..'
    id_column = 'id.layer_uuid_txt'  

    number_of_data = int(input("Enter the number of data to use for split: "))

    # Load configuration from JSON file
    with open(folder_path + '/config.json', 'r') as f:
        config = json.load(f)

    # File paths for testing datasets
    spectra_type = "mir"  # Adjust as needed
    log_targets = config.get('log_targets', [])  # Get list of log-transformed targets from config

    mir_data_path = './../datasets/ossl_mir_L0_v1.2.csv'
    soillab_path = './../datasets/ossl_soillab_L0_v1.2.csv'

    spectral_data = pd.read_csv(mir_data_path)
    soillab_data = pd.read_csv(soillab_path)

    merged_data = spectral_data.merge(soillab_data, on=id_column)

    evaluation_log_path = folder_path + '/evaluation_results/evaluation_log.csv'

    for target_column in config['targets']:
        # Loop over each target column in the configuration file
        print(f"Evaluating model for target: {target_column}")

        # Load the trained model
        is_logged = target_column in log_targets
        model_suffix = "_logged" if is_logged else ""
        model_path = f'{folder_path}/trained_models/plsr_model_{spectra_type}_{target_column}_{number_of_data}{model_suffix}.joblib'
        
        if not os.path.exists(model_path):
            print(f"Model file not found for target: {target_column}. Skipping evaluation...")
            continue

        print(f"Loading model from: {model_path}")
        model = joblib.load(model_path)

        # File paths for testing datasets for the current target
        train_test_path = f'{folder_path}/train_test_dataset/train_test_splits_{spectra_type}.json'

        # Load test data
        print(f"Loading test data for target: {target_column}")
        test_data = load_data(merged_data, train_test_path, id_column, target_column, number_of_data)

        if len(test_data) == 0:
            print("No test data found, skipping....")
            continue

        # Preprocess test data
        X_test, y_test = preprocess_data(test_data, target_column)

        # Make predictions on the testing data
        y_pred = model.predict(X_test)

        # Evaluate the model
        rmse, lccc, r2 = evaluate_model(y_test, y_pred)
        print(f"RMSE: {rmse:.4f}, LCCC: {lccc:.4f}, R²: {r2:.4f}")

        # Output the image of predicted vs actual values
        output_image_suffix = "_logged" if is_logged else ""
        output_image_path = f'{folder_path}/evaluation_results/{spectra_type}_{target_column}_evaluation_plsr_{number_of_data}{output_image_suffix}.png'
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        plot_evaluation(y_test, y_pred, rmse, lccc, r2, output_image_path, target_column)
        print(f"Evaluation image saved to: {output_image_path}")

        # Log evaluation results
        evaluation_result = {
            'model_used': 'PLSR',
            'parameters': model.get_params(),
            'no_of_samples': number_of_data,  # Adjust as needed
            'train_test_split_ratio': 0.2,  # Adjust as needed
            'target': target_column,
            'spectra_type': 'neo ' + spectra_type,
            'rmse': rmse,
            'lccc': lccc,
            'r_squared': r2,
            'logged': is_logged,
            'train_data_key': f'train_{target_column}_{number_of_data}',
            'test_data_key': f'test_{target_column}_{number_of_data}'
        }
        log_evaluation_results(evaluation_result, evaluation_log_path)
