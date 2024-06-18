import pandas as pd
from cubist import Cubist
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
import os
import json
import optuna

def load_data(train_data_path, test_data_path):
    # Load training and testing data from CSV files
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    return train_data, test_data

def evaluate_model(y_true, y_pred):
    # Evaluate the model using RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse


def preprocess_data(data, target_column):
    # Apply log1p to target data
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Apply log1p transformation to the target column
    y_log1p = np.log1p(y)

    return X, y_log1p

def optimize(train_data, test_data, target_column):
    # Define the objective function for hyperparameter optimization
    def objective(trial):
        # Sample a subset of the training data (e.g., 1000 samples)
        train_data_subset = train_data.sample(n=1000, random_state=42)

        # Define hyperparameters to optimize
        committees = trial.suggest_int('committees', 1, 20)
        neighbors = trial.suggest_int('neighbors', 1, 9)

        # Preprocess the data
        X_train_subset, y_train_subset = preprocess_data(train_data_subset, target_column)
        X_test, y_test = preprocess_data(test_data, target_column)

        # Initialize and train the Cubist model
        model = Cubist()
        model.neighbors = neighbors
        model.n_committees = committees
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

def train_cubist_model(train_data, test_data, target_column, best_params, output_model_path):
    # Preprocess the data
    X_train, y_train = preprocess_data(train_data, target_column)
    X_test, y_test = preprocess_data(test_data, target_column)

    # Initialize and train the Cubist model with the best hyperparameters
    model = Cubist()
    model.neighbors = best_params['neighbors']
    model.n_committees = best_params['committees']
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
    # Load configuration from JSON file
    with open('./../data_processing/config.json', 'r') as f:
        config = json.load(f)
    
    # File paths for training and testing datasets
    spectra_type = "mir"  # Adjust as needed

    for target_column in config['targets']:
        # Loop over each target column in the configuration file
        print(f"Training model for target: {target_column}")
            
        # File paths for training and testing datasets for the current target
        train_data_path = f'./../train_test_dataset/train_data_{spectra_type}_{target_column}.csv'
        test_data_path = f'./../train_test_dataset/test_data_{spectra_type}_{target_column}.csv'

        # Load data
        print(f"Loading data for target: {target_column}")
        train_data, test_data = load_data(train_data_path, test_data_path)

        ## sampling just 15k for model

        train_data = train_data.sample(n=10000, random_state=42)
            
        # Optimize hyperparameters
        print("Optimizing hyperparameters...")
        best_params = optimize(train_data, test_data, target_column)

        # Train and save the model with full data
        print(f"Training and saving model for target: {target_column}")
        print(f"Train data shape: {train_data.shape}, test data shape: {test_data.shape}")
        output_model_path = f'./../trained_models/cubist_model_{spectra_type}_{target_column}_logged.joblib'
        train_cubist_model(train_data, test_data, target_column, best_params, output_model_path)
