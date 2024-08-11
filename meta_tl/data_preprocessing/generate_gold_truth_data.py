import pandas as pd
import os

path_to_data_folder = '/Users/abdulnaser/Desktop/TL_Multi_Source_ER/data/'
path_to_help_data = path_to_data_folder + 'help_data/'
path_to_gt_folder = path_to_data_folder + 'ground_truth_data/'
path_to_sim_vectors_folder = path_to_data_folder + 'linkage_tasks'



def ground_truth_data_extractor():
    # Read data
    data = pd.read_csv(path_to_help_data + 'data_cleaned_very_new.csv')

    for filename in os.listdir(path_to_sim_vectors_folder):
        print(filename)
        if filename.endswith(".csv"):

            sim_vec_file_path = os.path.join(path_to_sim_vectors_folder, filename)

            # Read the sim_vec_csv file into a dataframe
            sim_vec_df = pd.read_csv(sim_vec_file_path)

            # Split the string by underscore
            parts = filename.split("_")

            # Extract the two parts
            first_source = parts[0]
            second_source = parts[1].replace('.csv','')


            first_source_df = data.loc[data['source'] == first_source]
            second_source_df = data.loc[data['source'] == second_source]

            # Merge DataFrames on the 'ID' column
            sim_vec_gt_df = pd.merge(first_source_df, second_source_df, on='recId', how='inner')  # 'inner' means use the intersection of keys from both frames

            sim_vec_gt_df = sim_vec_gt_df[['key_x','key_y','recId','recId']]

            # Rename the column names of the ground truth data to match the other columns in the other dataframe to be the merge succefull
            sim_vec_gt_df.rename(columns = {'key_x':'record_compared_1', 'key_y':'record_compared_2'},inplace=True)

            sim_vec_gt_df['is_match'] = 1

            # Merge the dataframes on 'record_compared_1' and 'record_compared_2'
            merged_df = pd.merge(sim_vec_df, sim_vec_gt_df, on = ['record_compared_1', 'record_compared_2'], how= 'left')

            # Fill NaN values in 'is_match' column with zeros
            merged_df['is_match'].fillna(0, inplace=True)

            sim_vec_gt_df.to_csv(path_to_gt_folder + filename)
            merged_df.to_csv(os.path.join(path_to_sim_vectors_folder, filename))


ground_truth_data_extractor()