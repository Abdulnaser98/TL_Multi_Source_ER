import os
import sys
import time
import warnings
import xgboost as xgb
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import pairwise_kernels

# Add the path to custom modules
sys.path.append('/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/')
from utils import prepare_dataframe_to_similarity_comparison, compute_weighted_mean_similarity

# Suppress specific warning messages
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)


def compute_mmd(X, Y, kernel='rbf', gamma=None):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two sets of samples.

    Parameters:
    X (ndarray): Array representing the first set of samples.
    Y (ndarray): Array representing the second set of samples.
    kernel (str or callable): Kernel function to use. Default is 'rbf' (Gaussian).
    gamma (float): Parameter for the RBF kernel. If None, it is inferred from data.

    Returns:
    float: The MMD value.
    """
    kernel = 'rbf' if kernel == 'squared_exp' else kernel
    K_XX = pairwise_kernels(X, X, metric=kernel, gamma=gamma)
    K_YY = pairwise_kernels(Y, Y, metric=kernel, gamma=gamma)
    K_XY = pairwise_kernels(X, Y, metric=kernel, gamma=gamma)

    m, n = X.shape[0], Y.shape[0]

    mmd_squared = (np.sum(K_XX) / (m * (m - 1)) -
                   2 * np.sum(K_XY) / (m * n) +
                   np.sum(K_YY) / (n * (n - 1)))

    return np.sqrt(mmd_squared)


def mmd_permutation_test(X, Y, num_permutations=100, **kwargs):
    """
    Perform a permutation test to assess the significance of MMD between two sets of samples.

    Parameters:
    X (ndarray): Array representing the first set of samples.
    Y (ndarray): Array representing the second set of samples.
    num_permutations (int): Number of permutations to perform.
    **kwargs: Additional arguments to pass to the MMD function.

    Returns:
    float: The p-value of the permutation test.
    """
    mmd_observed = compute_mmd(X, Y, **kwargs)
    combined = np.vstack([X, Y])
    n_samples1 = X.shape[0]

    greater_extreme_count = 0
    for _ in range(num_permutations):
        np.random.shuffle(combined)
        X_permuted, Y_permuted = combined[:n_samples1], combined[n_samples1:]
        mmd_permuted = compute_mmd(X_permuted, Y_permuted, **kwargs)
        if mmd_permuted >= mmd_observed:
            greater_extreme_count += 1

    return (greater_extreme_count + 1) / (num_permutations + 1)


def calculate_psi(old_results, new_results):
    """
    Computes the Population Stability Index (PSI) to measure the shift in distributions between two datasets.

    Parameters:
    old_results (array-like): Observed values from the original dataset.
    new_results (array-like): Observed values from the new dataset.

    Returns:
    float: The PSI value indicating the shift in distributions.
    """
    def psi(expected, actual):
        return np.sum((actual - expected) * np.log(actual / expected))

    old_expected = np.mean(old_results)
    new_expected = np.mean(new_results)

    return psi(old_expected, new_expected)


def compare_linkage_tasks(task_1_path, task_2_path, task_1_name, task_2_name, test_type, relevant_columns, stat_lists=None, multivariate=False):
    """
    Compares the distributions of two linkage tasks based on the specified test type.

    Parameters:
    task_1_path (str): Path to the first linkage task file.
    task_2_path (str): Path to the second linkage task file.
    task_1_name (str): Name of the first linkage task.
    task_2_name (str): Name of the second linkage task.
    test_type (str): The type of statistical test to perform ('ks_test', 'wasserstein_distance', 'calculate_psi', 'ML_based', or 'MMD').
    relevant_columns (list): List of relevant columns to compare.
    stat_lists (dict): Dictionary to store statistical test results for each column.
    multivariate (bool): Flag indicating if the test is multivariate.

    Returns:
    tuple: Names of the compared files and a list of resulting values from the statistical tests.
    """
    if multivariate:
        if test_type == 'ML_based':
            df_1 = prepare_dataframe_to_similarity_comparison(task_1_path)
            df_2 = prepare_dataframe_to_similarity_comparison(task_2_path)
            df_1['is_match'] = 0
            df_2['is_match'] = 1

            df_shuffled = pd.concat([df_1.sample(frac=1, random_state=42), df_2.sample(frac=1, random_state=42)], ignore_index=True)
            X, y = df_shuffled.drop(columns=['is_match']), df_shuffled['is_match']

            model = xgb.XGBClassifier(objective='binary:logistic')
            cv_score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()

            return task_1_name, task_2_name, cv_score

        elif test_type == 'MMD':
            df_1 = pd.read_csv(task_1_path)[relevant_columns].apply(pd.to_numeric, errors='coerce').fillna(2)
            df_2 = pd.read_csv(task_2_path)[relevant_columns].apply(pd.to_numeric, errors='coerce').fillna(2)
            mmd_value = mmd_permutation_test(df_1, df_2)
            return task_1_name, task_2_name, mmd_value

    else:
        df_1 = prepare_dataframe_to_similarity_comparison(task_1_path)
        df_2 = prepare_dataframe_to_similarity_comparison(task_2_path)
        intersection_columns = df_1.columns.intersection(df_2.columns)
        results = []

        for column in stat_lists:
            if column not in intersection_columns:
                stat_lists[column].append(-2)
            else:
                if test_type == 'ks_test':
                    ks_stat, p_value = ks_2samp(df_1[column], df_2[column])
                    stat_lists[column].append(p_value)
                    results.append(p_value)
                elif test_type == 'wasserstein_distance':
                    w_dist = wasserstein_distance(df_1[column], df_2[column])
                    stat_lists[column].append(w_dist)
                    results.append(w_dist)
                elif test_type == 'calculate_psi':
                    psi_value = calculate_psi(df_1[column], df_2[column])
                    stat_lists[column].append(psi_value)
                    results.append(psi_value)

        return task_1_name, task_2_name, results


def evaluate_similarity(results, test_type, case, alpha=0.05):
    """
    Evaluates the similarity between two files based on the test results.

    Parameters:
    results (list or float): Resulting value(s) from the statistical tests.
    test_type (str): The type of statistical test performed.
    case (int): Case number to determine the evaluation logic (1 or 2).
    alpha (float): Significance level for the ks_test (default is 0.05).

    Returns:
    int: 1 if the files are similar, 0 otherwise.
    """
    if test_type == 'ML_based':
        return 0 if results > 0.80 else 1

    elif test_type == 'MMD':
        return 0 if results < 0.05 else 1

    elif isinstance(results, list):
        if case == 1:  # All features should have the same distribution
            if test_type == 'ks_test':
                return 0 if any(value < alpha for value in results) else 1
            else:  # 'wasserstein_distance' or 'calculate_psi'
                return 0 if any(value > 0.1 for value in results) else 1
        else:  # Majority of features should have the same distribution
            threshold = (lambda x: x > alpha) if test_type == 'ks_test' else (lambda x: x < 0.1)
            similar_count = sum(threshold(value) for value in results)
            return 1 if similar_count >= len(results) // 2 else 0


def compute_similarity_test(case, test_type, tasks_path, relevant_columns, multivariate=False):
    """
    Computes the similarity between pairs of record linkage tasks using various statistical tests.

    Parameters:
    case (int): Determines the logic for evaluating similarity (1 or 2).
    test_type (str): The type of statistical test to perform.
    tasks_path (str): Path to the folder containing the record linkage tasks.
    relevant_columns (list): List of relevant columns to compare.
    multivariate (bool): Flag indicating if the test is multivariate.

    Returns:
    tuple: A DataFrame containing similar record linkage tasks and a general DataFrame with all comparisons.
    """
    stat_lists = {col: [] for col in relevant_columns} if not multivariate else None

    first_tasks, second_tasks, similarities, processed_pairs = [], [], [], []
    alpha = 0.05

    task_files = [file for file in os.listdir(tasks_path) if file.endswith('.csv')]

    for task_1 in task_files:
        task_1_path = os.path.join(tasks_path, task_1)
        for task_2 in task_files:
            task_2_path = os.path.join(tasks_path, task_2)
            pair_identifier = f"{task_1}_{task_2}"
            reverse_pair = f"{task_2}_{task_1}"

            if task_1_path != task_2_path and pair_identifier not in processed_pairs and reverse_pair not in processed_pairs:
                processed_pairs.append(pair_identifier)

                file1, file2, results = compare_linkage_tasks(
                    task_1_path, task_2_path, task_1, task_2, test_type, relevant_columns, stat_lists, multivariate
                )
                similarity = evaluate_similarity(results, test_type, case, alpha)

                first_tasks.append(file1)
                second_tasks.append(file2)
                similarities.append(similarity)

    results_df = pd.DataFrame({
        'first_task': first_tasks,
        'second_task': second_tasks,
        **stat_lists,
        'similarity': similarities
    }) if not multivariate else pd.DataFrame({
        'first_task': first_tasks,
        'second_task': second_tasks,
        'similarity': similarities
    })

    similar_tasks_df = results_df[results_df['similarity'] == 1]

    if not multivariate:
        results_df['avg_similarity'] = results_df.apply(lambda row: compute_weighted_mean_similarity(row[2:-1]), axis=1)
        similar_tasks_df['avg_similarity'] = similar_tasks_df.apply(lambda row: compute_weighted_mean_similarity(row[2:-1]), axis=1)

    # Output statistics
    num_similar_tasks = similar_tasks_df.shape[0]
    print(f"Number of tasks with similar distribution: {num_similar_tasks}")

    unique_first_tasks = similar_tasks_df['first_task'].unique()
    unique_second_tasks = similar_tasks_df['second_task'].unique()

    unique_tasks_count = len(set(unique_first_tasks) | set(unique_second_tasks))
    print(f"Total number of unique tasks with similar distribution: {unique_tasks_count}")

    return similar_tasks_df, results_df
