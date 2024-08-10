import os
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score

def evaluate_predictions(predictions_folder):
    """
    Evaluate the performance of active learning predictions by calculating F1 score, precision, and recall.

    Parameters:
    predictions_folder (str): The path to the folder containing the prediction CSV files.

    Returns:
    None
    """
    # List to store individual DataFrames from all prediction files
    dataframes = []

    # Iterate through each file in the predictions folder
    for filename in os.listdir(predictions_folder):
        if filename.endswith('.csv'):
            # Construct the full path to the file
            filepath = os.path.join(predictions_folder, filename)
            # Read the CSV file into a DataFrame
            df = pd.read_csv(filepath)

            # Modify the record comparison columns to include the file identifiers
            file_identifiers = filename.split('_')
            df['record_compared_1'] = df['record_compared_1'] + "_" + file_identifiers[0]
            df['record_compared_2'] = df['record_compared_2'] + "_" + file_identifiers[1]

            # Append the DataFrame to the list
            dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Fill missing predictions with the actual labels
    combined_df.loc[combined_df['pred'].isna(), 'pred'] = combined_df.loc[combined_df['pred'].isna(), 'is_match']

    # Output the size of the combined DataFrame
    print(f"Combined DataFrame shape: {combined_df.shape[0]}")

    # Calculate and print the evaluation metrics
    f1 = f1_score(combined_df['is_match'], combined_df['pred'])
    precision = precision_score(combined_df['is_match'], combined_df['pred'])
    recall = recall_score(combined_df['is_match'], combined_df['pred'])

    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

# Example usage (assuming the predictions folder path is defined):
# evaluate_predictions('/path/to/predictions/folder')
