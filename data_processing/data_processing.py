import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def merge_data(main_data, soillab_data, soilsite_data):
    merged_data = main_data.merge(soillab_data, on='id.layer_uuid_txt').merge(soilsite_data, on='id.layer_uuid_txt')
    return merged_data


def clean_data(data, target_columns):

    # Only keep target columns and spectral columns in the dataframe
    spectral_columns = [col for col in data.columns if 'scan_mir' in col or 'scan_visnir' in col]
    relevant_columns = spectral_columns + target_columns
    relevant_data = data[relevant_columns]
    cleaned_data = relevant_data.dropna(subset=spectral_columns)

    return cleaned_data


def normalize_data(data, target_columns):
    spectral_columns = [col for col in data.columns if col not in target_columns]
    
    for col in spectral_columns:
        mean = data[col].mean()
        std = data[col].std()
        
        data[col] = (data[col] - mean) / std
    
    return data

def split_data(data,targets, target_column, test_size=0.2, random_state=42):

    data = data.dropna(subset=[target_column])
    
    q_low = data[target_column].quantile(0.1)
    q_hi  = data[target_column].quantile(0.9)
    data_filtered = data[(data[target_column] < q_hi) & (data[target_column] > q_low)]

    # Drop other target columns except target_column
    other_targets = [col for col in targets if col != target_column]
    data_filtered = data_filtered.drop(columns=other_targets)

    train_data, test_data = train_test_split(data_filtered, test_size=test_size, random_state=random_state)
    
    return train_data, test_data



def process_data(spectra_path, soillab_path, soilsite_path, processed_data_path, train_test_dir, dataset_type, target_columns):
   
    # Load the data
    print(f"Loading the {dataset_type} data...")
    raw_data = load_data(spectra_path)
    print(f"Data loaded successfully with shape: {raw_data.shape}")
    
    # Load soillab and soilsite data
    print("Loading soillab and soilsite data...")
    soillab_data = load_data(soillab_path)
    soilsite_data = load_data(soilsite_path)
    print(f"Soillab data loaded successfully with shape: {soillab_data.shape}")
    print(f"Soilsite data loaded successfully with shape: {soilsite_data.shape}")

    # Merge the data
    print(f"Merging the {dataset_type} data with soillab and soilsite data...")
    merged_data = merge_data(raw_data, soillab_data, soilsite_data)
    print(f"Data merged successfully. New shape: {merged_data.shape}")

    # Clean the data
    print(f"Cleaning the merged {dataset_type} data...")
    cleaned_data = clean_data(merged_data, target_columns)
    print(f"Data cleaned successfully. New shape: {cleaned_data.shape}")

    
    # Normalize the data
    print(f"Normalizing the {dataset_type} data...")
    final_data = normalize_data(cleaned_data,target_columns)
    print(f"Data normalization completed.")

    # Save the processed data
    final_data.to_csv(processed_data_path, index=False)
    print(f"Processed {dataset_type} data saved to {processed_data_path}")

    # Split the data into training and testing sets for each target
    for target_column in target_columns:
        
        print(f"Splitting {dataset_type} data for target: {target_column}")
        train_data, test_data = split_data(final_data, target_columns, target_column)
        
        # Save the split data
        train_data.to_csv(f'{train_test_dir}/train_data_{dataset_type}_{target_column}.csv', index=False)
        test_data.to_csv(f'{train_test_dir}/test_data_{dataset_type}_{target_column}.csv', index=False)
        print(f"Saved {dataset_type} train-test split for target: {target_column}")

    print(f"All {dataset_type} data processing steps completed successfully.")

if __name__ == "__main__":
    soillab_path = './../datasets/ossl_soillab_L0_v1.2.csv'
    soilsite_path = './../datasets/ossl_soilsite_L0_v1.2.csv'
    train_test_dir = './../train_test_dataset'


    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    target_columns = config['targets']
    
    # Define file paths for MIR data
    mir_data_path = './../datasets/ossl_mir_L0_v1.2.csv'
    mir_processed_data_path = './../datasets/processed_data_mir.csv'
    
    # Process MIR data
    process_data(mir_data_path, soillab_path, soilsite_path, mir_processed_data_path, train_test_dir, 'mir', target_columns)

    # Define file paths for VisNIR data
    visnir_data_path = './../datasets/ossl_visnir_L0_v1.2.csv'
    visnir_processed_data_path = './../datasets/processed_data_visnir.csv'
    
    # Process VisNIR data
    process_data(visnir_data_path, soillab_path, soilsite_path, visnir_processed_data_path, train_test_dir, 'visnir', target_columns)