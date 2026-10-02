import pandas as pd
import random

def balance_csv(file_path, output_path):
    # Load the CSV into a DataFrame
    df = pd.read_csv(file_path)

    # Ensure the label column exists
    if "label" not in df.columns:
        raise ValueError("The CSV file must contain a 'label' column.")

    # Group by the label column
    groups = df.groupby("label")

    # Find the minimum number of rows among all labels (0-9)
    min_count = min(len(groups.get_group(label)) for label in range(10))

    # Balance the dataset
    balanced_dfs = []
    for label in range(10):
        group = groups.get_group(label)
        balanced_dfs.append(group.sample(n=min_count, random_state=42))

    # Concatenate all balanced groups
    balanced_df = pd.concat(balanced_dfs)

    # Shuffle the final balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to a new CSV file
    balanced_df.to_csv(output_path, index=False)
    print(f"The balanced dataset has been saved to {output_path}.")

# Usage example
input_csv = "MNIST dataset\mnist_train.csv"  # Replace with your input CSV file path
output_csv = "MNIST dataset\mnist_train1.csv"  # Replace with your desired output file path
balance_csv(input_csv, output_csv)