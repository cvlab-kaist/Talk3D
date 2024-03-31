import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def load_data(csv_path, columns_to_read):
    data = pd.read_csv(csv_path)
    selected_data = data[columns_to_read]
    return selected_data

def compute_mse_loss(data1, data2):
    if len(data1) != len(data2):
        shorter_length = min(len(data1), len(data2))
        data1 = data1[:shorter_length]
        data2 = data2[:shorter_length]
    mse = mean_squared_error(data1, data2)
    return mse

def compute_aue_from_csv(csv1, csv2):
    # strict_columns_to_read = [" AU10_r", " AU14_r", " AU20_r", " AU25_r", " AU26_r"]
    # columns_to_read = [" AU09_r", " AU10_r", " AU12_r", " AU14_r", " AU15_r", " AU17_r", " AU20_r", " AU23_r", " AU25_r", " AU26_r"]
    # strict_columns_to_read_c = [" AU10_c", " AU14_c", " AU20_c", " AU25_c", " AU26_c"]
    columns_to_read_c = [" AU09_c", " AU10_c", " AU12_c", " AU14_c", " AU15_c", " AU17_c", " AU20_c", " AU23_c", " AU25_c", " AU26_c"]
        
    AUE_strict_C, AUE_C = 0, 0
    
    for column in columns_to_read_c:
        data1 = load_data(csv1, column)
        data2 = load_data(csv2, column)
        AUE_C += compute_mse_loss(data1, data2)
        
    return float(AUE_C)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MSE loss between two CSV files")
    parser.add_argument("csv_path1", help="Path to the first CSV file")
    parser.add_argument("csv_path2", help="Path to the second CSV file")
    args = parser.parse_args()

    # Columns to read from the CSV files
    strict_columns_to_read = [" AU10_r", " AU14_r", " AU20_r", " AU25_r", " AU26_r"]
    columns_to_read = [" AU09_r", " AU10_r", " AU12_r", " AU14_r", " AU15_r", " AU17_r", " AU20_r", " AU23_r", " AU25_r", " AU26_r"]
    strict_columns_to_read_c = [" AU10_c", " AU14_c", " AU20_c", " AU25_c", " AU26_c"]
    columns_to_read_c = [" AU09_c", " AU10_c", " AU12_c", " AU14_c", " AU15_c", " AU17_c", " AU20_c", " AU23_c", " AU25_c", " AU26_c"]

    # Load data from CSV files
    AUE_strict, AUE = 0, 0
    for column in strict_columns_to_read:
        # print(f"Reading column: {column}")
        data1 = load_data(args.csv_path1, column)
        data2 = load_data(args.csv_path2, column)
        AUE_strict += compute_mse_loss(data1, data2)
    # mse_loss = compute_mse_loss(data1, data2)

    for column in columns_to_read:
        # print(f"Reading column: {column}")
        data1 = load_data(args.csv_path1, column)
        data2 = load_data(args.csv_path2, column)
        AUE += compute_mse_loss(data1, data2)
        
    AUE_strict_C, AUE_C = 0, 0
    for column in strict_columns_to_read_c:
        # print(f"Reading column: {column}")
        data1 = load_data(args.csv_path1, column)
        data2 = load_data(args.csv_path2, column)
        AUE_strict_C += compute_mse_loss(data1, data2)
    
    for column in columns_to_read_c:
        # print(f"Reading column: {column}")
        data1 = load_data(args.csv_path1, column)
        data2 = load_data(args.csv_path2, column)
        AUE_C += compute_mse_loss(data1, data2)
    
    print(f"AUE_strict: {AUE_strict}")
    print(f"AUE: {AUE}")
    print(f"AUE_strict_C: {AUE_strict_C}")
    print(f"AUE_C: {AUE_C}")