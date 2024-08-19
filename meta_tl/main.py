# Record the start time for performance measurement
import time
start_time = time.time()
import os
import sys
import warnings

import pandas as pd

# Ignore all warnings
warnings.filterwarnings("ignore")

# Define the main path for the project
MAIN_PATH = '/Users/abdulnaser/Desktop/TL_Multi_Source_ER/'

# Add the path to the custom module directory
sys.path.append(MAIN_PATH)

# Local application imports
from utils import (
    remove_empty_files,
    count_files_in_directory,
    count_total_records,
)
from statistical_tests import compute_similarity_test
from graph_clustering import create_graph, detect_communities
from model_selection import select_linkage_tasks_from_communities
from active_learning import label_linkage_tasks
from tranfear_learning_process import transfear_learning_process_main
from evaluation import evaluate_predictions

# Define paths to various data directories
RECORD_LINKAGE_TASKS_PATH = os.path.join(MAIN_PATH, 'data/linkage_tasks/')
CLEANED_DATA_SOURCES_PATH = os.path.join(MAIN_PATH, 'data/cleaned_data/')
LABELED_RECORD_LINKAGE_TASKS_PATH = os.path.join(MAIN_PATH, 'data/linkage_tasks_labeled/')


# Define the configuration parameters
STATISTICAL_TEST = 'wasserstein_distance' # or
FEATURE_CASE = 2  # 1 for all features to have the same distribution, 2 for majority of features to have the same distributions
COMMUNITY_DETECTION_ALGORITHM = 'girvan_newman'  # or 'label_propagation_clustering'
ACTIVE_LEARNING_ALGORITHM = 'bootstrapping'  # or 'margin'
ACTIIVE_LEARNING_MIN_BUDGET = 20
ACTIVE_LEARNING_ITERATION_BUDGET = 20
ACTIVE_LEARNING_TOTAL_BUDGET = 2000

RELEVANT_COLUMNS_IN_LINKAGE_TASKS = ['Produktname_dic3', 'Modell_Liste_3g', 'MPN_Liste_TruncateBegin20',
            'EAN_Liste_TruncateBegin20', 'Digital_zoom_NumMaxProz30',
            'optischer_zoom_NumMaxProz30', 'Breite_NumMaxProz30', 'Höhe_NumMaxProz30',
            'Gewicht_NumMaxProz30', 'Sensortyp_Jaccard3']

RELEVANT_COLUMNS_IN_ACTIVE_LEARNING = ['record_compared_1','record_compared_2','MPN_Liste_TruncateBegin20','EAN_Liste_TruncateBegin20','Produktname_dic3',
               'Modell_Liste_3g','Digital_zoom_NumMaxProz30','optischer_zoom_NumMaxProz30',
               'Breite_NumMaxProz30', 'Höhe_NumMaxProz30', 'Gewicht_NumMaxProz30', 'Sensortyp_Jaccard3','is_match']

# ===================================================
# Step 1: Prepare Record Linkage Tasks
# ===================================================

# Remove empty record linkage tasks
remove_empty_files(RECORD_LINKAGE_TASKS_PATH)

# Count and print the number of non-empty record linkage tasks
num_linkage_tasks = count_files_in_directory(RECORD_LINKAGE_TASKS_PATH)
print(f"Number of non-empty record linkage tasks: {num_linkage_tasks}")

# Count and print the number of unique data sources in the cleaned data
num_cleaned_data_sources = count_files_in_directory(CLEANED_DATA_SOURCES_PATH)
print(f"Number of unique cleaned data sources: {num_cleaned_data_sources}")

# Count and print the total number of record pairs across all tasks
total_record_pairs = count_total_records(RECORD_LINKAGE_TASKS_PATH)
print(f"Total number of record pairs: {total_record_pairs}")

# ===================================================
# Step 2: Perform Linkage Tasks Distribution Test
# ===================================================

# Record the start time for performance measurement
start_time = time.time()

# Compute the similarity test based on the chosen statistical test and case
linkage_tasks_similarity_df, linkage_tasks_general_df = compute_similarity_test(
    FEATURE_CASE, STATISTICAL_TEST, RECORD_LINKAGE_TASKS_PATH,RELEVANT_COLUMNS_IN_LINKAGE_TASKS
)

# Calculate and print the elapsed time for the similarity test
elapsed_time = time.time() - start_time
print(f"Elapsed time for similarity test: {elapsed_time:.2f} seconds")


# ===================================================
# Step 3: Perform Graph Clustering on Linkage Tasks
# ===================================================

# Create a graph of linkage tasks based on their similarity
graph, task_mapping = create_graph(linkage_tasks_similarity_df)

# Record the start time for graph clustering
start_time = time.time()

# Detect communities within the graph using the selected algorithm
linkage_task_communities = detect_communities(
    COMMUNITY_DETECTION_ALGORITHM, graph
)

# Calculate and print the elapsed time for graph clustering
elapsed_time = time.time() - start_time
print(f"Elapsed time for graph clustering: {elapsed_time:.2f} seconds")

# ===================================================
# Step 4: Select Linkage Tasks from Each Community
# ===================================================

# Record the start time for task selection
start_time = time.time()

# Select the largest task from each community for model training
selected_tasks = select_linkage_tasks_from_communities(
    RECORD_LINKAGE_TASKS_PATH, linkage_task_communities, task_mapping
)

# Calculate and print the elapsed time for task selection
elapsed_time = time.time() - start_time
print(f"Elapsed time for task selection: {elapsed_time:.2f} seconds")

# ===================================================
# Step 5: Apply Active Learning to Label Selected Tasks
# ===================================================

# Record the start time for active learning
start_time = time.time()

# Apply active learning to label the selected tasks (uncomment when ready)
labeled_tasks = label_linkage_tasks(
     selected_tasks, RECORD_LINKAGE_TASKS_PATH, linkage_tasks_general_df,
     min_budget=ACTIIVE_LEARNING_MIN_BUDGET,iteration_budget=ACTIVE_LEARNING_ITERATION_BUDGET, total_budget=ACTIVE_LEARNING_TOTAL_BUDGET, labeled_tasks_dir=LABELED_RECORD_LINKAGE_TASKS_PATH,active_learning_strategy = ACTIVE_LEARNING_ALGORITHM,
     relevant_columns = RELEVANT_COLUMNS_IN_ACTIVE_LEARNING
)

# Calculate and print the elapsed time for active learning
elapsed_time = time.time() - start_time
print(f"Elapsed time for active learning: {elapsed_time:.2f} seconds")

# ===================================================
# Step 6: Perform Transfer Learning and Inference
# ===================================================
# Record the start time for active learning
start_time = time.time()

# Perform the main transfer learning process on the selected tasks
transfear_learning_process_main(
    selected_tasks, LABELED_RECORD_LINKAGE_TASKS_PATH,
    RECORD_LINKAGE_TASKS_PATH,RELEVANT_COLUMNS_IN_ACTIVE_LEARNING
)

# Calculate and print the elapsed time for active learning
elapsed_time = time.time() - start_time
print(f"Elapsed time for active learning: {elapsed_time:.2f} seconds")

# ===================================================
# Step 7: Evaluate the Results
# ===================================================
# Record the start time for active learning
start_time = time.time()

# Evaluate the performance of the labeled tasks
evaluate_predictions(LABELED_RECORD_LINKAGE_TASKS_PATH)

# Calculate and print the elapsed time for active learning
elapsed_time = time.time() - start_time
print(f"Elapsed time for active learning: {elapsed_time:.2f} seconds")


# Calculate and print the elapsed time for the similarity test
elapsed_time = time.time() - start_time
print(f"Final Elapsed Time for the entire project is: {elapsed_time:.2f} seconds")

