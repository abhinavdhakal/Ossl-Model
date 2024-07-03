import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def merge_soillab_data(main_data, soillab_data, id_column):
    merged_data = main_data.merge(soillab_data, on=id_column)
    return merged_data

def clean_data(data, target_columns, id_column):
    # Only keep target columns and spectral columns in the dataframe
    spectral_columns = [col for col in data.columns if 'scan_mir' in col or 'scan_visnir' in col or 'scan_nir' in col]

    relevant_columns = spectral_columns + target_columns + [id_column]
    relevant_data = data[relevant_columns]
    cleaned_data = relevant_data.dropna(subset=spectral_columns)

    return cleaned_data


def normalize_data(data, target_columns, id_column):
    spectral_columns = [col for col in data.columns if 'scan_mir' in col or 'scan_visnir' in col or 'scan_nir' in col]
    for col in spectral_columns:
        mean = data[col].mean()
        std = data[col].std()
        data[col] = (data[col] - mean) / std
    
    return data


def split_data(data, target_columns, id_column, test_size=0.2, no_of_data=None):
    all_splits = {}

    for target_column in target_columns:
        data_target = data.dropna(subset=[target_column])

        q_low = data_target[target_column].quantile(0.1)
        q_hi = data_target[target_column].quantile(0.9)
        data_filtered = data_target[(data_target[target_column] < q_hi) & (data_target[target_column] > q_low)]

        # Check if no_of_data is larger than available data
        if no_of_data is not None and no_of_data > len(data_filtered):
            print(f"Warning: Requested number of data ({no_of_data}) is larger than available data for target {target_column}. Using available data instead.")
            new_no_of_data = len(data_filtered)  # Cap no_of_data to available data
        else:
            new_no_of_data = no_of_data

        if(new_no_of_data < 100):
            print("Empty dataset for: ", data_target)
            continue

        data_filtered = data_filtered.sample(new_no_of_data, random_state=42)  # Shuffle data

        train_data, test_data = train_test_split(data_filtered, test_size=test_size, random_state=42)

        key = f'{target_column}_{new_no_of_data if new_no_of_data is not None else "all"}'

        # Add the split data to the dictionary
        all_splits[f'train_{key}'] = train_data[id_column].tolist()
        all_splits[f'test_{key}'] = test_data[id_column].tolist()

    return all_splits

def process_data(spectra_path, soillab_path, train_test_dir, dataset_type, target_columns, id_column, no_of_data):
    # Load the spectral data
    print(f"Loading the {dataset_type} data...")
    raw_data = load_data(spectra_path)
    print(f"Data loaded successfully with shape: {raw_data.shape}")

    # Load the soillab data
    print("Loading soillab data...")
    soillab_data = load_data(soillab_path)
    print(f"Soillab data loaded successfully with shape: {soillab_data.shape}")

    # Merge the spectral data with the soillab data
    print(f"Merging the {dataset_type} data with soillab data...")
    merged_data = merge_soillab_data(raw_data, soillab_data, id_column)
    print(f"Data merged successfully. New shape: {merged_data.shape}")

    # Clean the data
    print(f"Cleaning the {dataset_type} data...")
    cleaned_data = clean_data(merged_data, target_columns, id_column)
    print(f"Data cleaned successfully. New shape: {cleaned_data.shape}")

    
    # Normalize the data
    print(f"Normalizing the {dataset_type} data...")
    final_data = normalize_data(cleaned_data, target_columns, id_column)
    print(f"Data normalization completed.")

    # Prepare the JSON file path
    json_file_path = f'{train_test_dir}/train_test_splits_{dataset_type}_neo.json'
    
    # Load existing data if the file exists
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            all_splits = json.load(json_file)
    else:
        all_splits = {}

    # Split the data into training and testing sets
    print(f"Splitting {dataset_type} data...")
    new_splits = split_data(final_data, target_columns, id_column, test_size=0.2, no_of_data=no_of_data)    
    
    # Update all_splits with new_splits
    all_splits.update(new_splits)

    # Save all splits back to the JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(all_splits, json_file, indent=4)

    print(f"All {dataset_type} data processing steps completed successfully.")

if __name__ == "__main__":
    folder_path = './..'

    train_test_dir = folder_path + '/train_test_dataset'
    os.makedirs(train_test_dir, exist_ok=True)

    # Load configuration
    with open(folder_path + '/config.json', 'r') as f:
        config = json.load(f)
    
    target_columns = config['targets']
    id_column = 'id.sample_local_c'  # Column name to use for saving train-test split

    # Ask user for number of data to use for the split
    number_of_data = int(input("Enter the number of data to use for split: "))

    # Define file paths for MIR data
    mir_data_path = folder_path + '/datasets/neospectra_mir_v1.2.csv'
    soillab_path = folder_path + '/datasets/neospectra_soillab_v1.2.csv'
    
    # Process MIR data
    process_data(mir_data_path, soillab_path, train_test_dir, 'mir', target_columns, id_column, number_of_data)

    # Define file paths for VisNIR data
    nir_data_path = folder_path + '/datasets/neospectra_nir_v1.2.csv'
    
    # Process VisNIR data
    process_data(nir_data_path, soillab_path, train_test_dir, 'nir', target_columns, id_column, number_of_data)
