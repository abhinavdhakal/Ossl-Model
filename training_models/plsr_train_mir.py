import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
import joblib
import os
import json
import optuna

def load_data(merged_data, train_test_path, id_column, target_column, number_of_data):
    with open(train_test_path, 'r') as json_file:
        all_splits = json.load(json_file)

    key = f'{target_column}_{number_of_data}'
    train_ids = all_splits[f'train_{key}']
    test_ids = all_splits[f'test_{key}']

    print(f"Train ids : {len(train_ids)}")
    print(f"Test ids : {len(test_ids)}")

    train_data = merged_data[merged_data[id_column].isin(train_ids)]
    test_data = merged_data[merged_data[id_column].isin(test_ids)]

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    return train_data, test_data

def evaluate_model(y_true, y_pred):
    # Evaluate the model using RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

def preprocess_data(data, target_column):
    spectral_columns = [col for col in data.columns if 'scan_mir' in col or 'scan_visnir' in col or 'scan_nir' in col]

    X = data[spectral_columns]
    y = data[target_column]
    
    if target_column in log_targets:
        y = np.log1p(y)

    return X, y

def optimize(train_data, test_data, target_column):
    # Define the objective function for hyperparameter optimization
    def objective(trial):
        # Sample a subset of the training data
        train_data_subset = train_data.sample(n=1000, random_state=42)

        # Define hyperparameters to optimize
        n_components = trial.suggest_int('n_components', 2, min(train_data.shape[1], 20))

        # Preprocess the data
        X_train_subset, y_train_subset = preprocess_data(train_data_subset, target_column)
        X_test, y_test = preprocess_data(test_data, target_column)

        # Initialize and train the PLSR model
        model = PLSRegression(n_components=n_components)
        model.fit(X_train_subset, y_train_subset)

        # Make predictions on the testing data
        y_pred = model.predict(X_test)

        # Evaluate the model using RMSE
        rmse = evaluate_model(y_test, y_pred)
        return rmse

    # Set up Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    # Get the best hyperparameters
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)
    return best_params

def train_plsr_model(train_data, test_data, target_column, best_params, output_model_path):
    # Preprocess the data
    X_train, y_train = preprocess_data(train_data, target_column)
    X_test, y_test = preprocess_data(test_data, target_column)

    # Initialize and train the PLSR model with the best hyperparameters
    model = PLSRegression(n_components=best_params['n_components'])
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the model
    rmse = evaluate_model(y_test, y_pred)
    print("Root Mean Squared Error (RMSE):", rmse)

    # Save the trained model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(model, output_model_path)
    print(f"Trained model saved to: {output_model_path}")

if __name__ == "__main__":
    folder_path = './..'
    id_column = 'id.layer_uuid_txt'

    # Load configuration from JSON file
    with open(folder_path + '/config.json', 'r') as f:
        config = json.load(f)
    
    # File paths for training and testing datasets
    spectra_type = "mir"  # Adjust as needed
    log_targets = config.get('log_targets', [])  # Get list of log-transformed targets from config

    mir_data_path = './../datasets/ossl_mir_L0_v1.2.csv'
    soillab_path = './../datasets/ossl_soillab_L0_v1.2.csv'

    spectral_data = pd.read_csv(mir_data_path)
    soillab_data = pd.read_csv(soillab_path)
    merged_data = spectral_data.merge(soillab_data, on=id_column)

    train_test_path = './../train_test_dataset/train_test_splits_mir.json'

    number_of_data = int(input("Enter the number of data to use for split: "))

    for target_column in config['targets']:
        # Loop over each target column in the configuration file
        print(f"Training model for target: {target_column}")
            
        # File paths for training and testing datasets for the current target

        # Check if data files exist
        if not (os.path.exists(train_test_path)):
            print(f"Data files not found for target: {target_column}. Skipping...")
            continue

        # Load data
        print(f"Loading data for target: {target_column}")
        train_data, test_data = load_data(merged_data, train_test_path, id_column, target_column, number_of_data)

        # Check if there are enough samples available
        if len(train_data) == 0:
            print(f"No training data available for target: {target_column}. Skipping...")
            continue

        # Optimize hyperparameters
        print("Optimizing hyperparameters...")
        best_params = optimize(train_data, test_data, target_column)

        # Train and save the model with full data
        print(f"Training and saving model for target: {target_column}")
        print(f"Train data shape: {train_data.shape}, test data shape: {test_data.shape}")
        
        output_model_suffix = "_logged" if target_column in log_targets else ""
        output_model_path = f'{folder_path}/trained_models/plsr_model_{spectra_type}_{target_column}_{number_of_data}{output_model_suffix}.joblib'

        train_plsr_model(train_data, test_data, target_column, best_params, output_model_path)
