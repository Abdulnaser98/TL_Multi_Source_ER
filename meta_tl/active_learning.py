import os
import sys
import time
import math
import operator
import numpy as np
import pandas as pd
import xgboost as xgb

# Add the path to custom modules
sys.path.append('/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/')
from utils import delete_files, prepare_dataframe_prediction


def allocate_active_learning_budget(selected_tasks, tasks_dir, tasks_info_df, min_budget_per_task, total_budget):
    """
    Allocate the active learning budget to both community (non-singleton) and singleton tasks.

    Parameters:
    selected_tasks (dict): A dictionary of selected tasks.
    tasks_dir (str): The directory containing the linkage tasks files.
    tasks_info_df (pd.DataFrame): DataFrame containing information about linkage tasks.
    min_budget_per_task (int): The minimum budget to allocate per task.
    total_budget (int): The total budget available for allocation.

    Returns:
    dict: A dictionary with task names as keys and their allocated budgets as values.
    """
    budget_allocation = {}

    # Allocate budget for community (non-singleton) tasks
    for task in selected_tasks:
        task_path = os.path.join(tasks_dir, task)
        task_df = pd.read_csv(task_path)
        budget_allocation[task] = task_df.shape[0]

    # Identify singleton tasks (tasks that do not belong to any community)
    non_singleton_tasks = tasks_info_df[tasks_info_df['similarity'] == 1]['first_task'].tolist()
    all_tasks = set(tasks_info_df['first_task'].unique()).union(tasks_info_df['second_task'].unique())

    # Allocate budget for singleton tasks
    for task in all_tasks:
        if task not in non_singleton_tasks:
            task_path = os.path.join(tasks_dir, task)
            task_df = pd.read_csv(task_path)
            budget_allocation[task] = task_df.shape[0]

    # Calculate the total minimum budget required
    total_min_budget = min_budget_per_task * len(budget_allocation)

    # Ensure the total minimum budget does not exceed the total available budget
    if total_min_budget > total_budget:
        raise ValueError("The total minimum budget exceeds the total available budget.")

    # Calculate the remaining budget after allocating the minimum budget
    remaining_budget = total_budget - total_min_budget

    # Calculate the proportional allocations for the remaining budget
    total_size = sum(budget_allocation.values())
    proportional_allocations = {
        task: (size / total_size) * remaining_budget for task, size in budget_allocation.items()
    }

    # Combine the minimum budget with the proportional allocations and round up the final allocations
    final_allocations = {
        task: math.ceil(min_budget_per_task + budget) for task, budget in proportional_allocations.items()
    }

    return final_allocations


def active_learning_bootstrap(task_df, iteration_budget, total_budget, num_classifiers=5):
    """
    Apply active learning using bootstrapping to label the records in the linkage task.

    Parameters:
    task_df (pd.DataFrame): DataFrame containing the task data.
    iteration_budget (int): The budget for each iteration.
    total_budget (int): The total budget available.
    num_classifiers (int): The number of classifiers to use for bootstrapping.

    Returns:
    pd.DataFrame: The labeled DataFrame.
    int: The total budget used.
    """
    start_time = time.time()

    # Initial training data selection using a random sample
    seed_indices = np.random.choice(task_df.shape[0], iteration_budget, replace=False)
    training_df = task_df.iloc[seed_indices]

    # Ensure diverse initial training set by checking for label diversity
    while len(training_df['is_match'].unique()) <= 1:
        seed_indices = np.random.choice(task_df.shape[0], iteration_budget, replace=False)
        training_df = task_df.iloc[seed_indices]

    # Create the unlabeled dataset by removing the selected training data
    unlabeled_df = task_df.drop(seed_indices)
    used_budget = training_df.shape[0]

    # Continue the active learning process until the budget is exhausted
    while used_budget < total_budget:
        classifiers = []

        # Train multiple classifiers using bootstrapping
        for _ in range(num_classifiers):
            clf = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
            bootstrap_indices = np.random.choice(training_df.shape[0], len(training_df), replace=True)
            bootstrap_df = training_df.iloc[bootstrap_indices]

            # Ensure label diversity in the bootstrapped dataset
            while len(bootstrap_df['is_match'].unique()) <= 1:
                bootstrap_indices = np.random.choice(training_df.shape[0], len(training_df), replace=True)
                bootstrap_df = training_df.iloc[bootstrap_indices]

            # Train the classifier on the bootstrapped dataset
            X_train = bootstrap_df.iloc[:, 2:-1].apply(pd.to_numeric, errors='coerce')
            y_train = bootstrap_df['is_match']
            clf.fit(X_train, y_train)
            classifiers.append(clf)

        # Calculate uncertainty for each unlabeled example using the ensemble of classifiers
        uncertainties = {
            idx: np.mean([clf.predict([record.iloc[2:-1]])[0] for clf in classifiers]) * (1 - np.mean([clf.predict([record.iloc[2:-1]])[0] for clf in classifiers]))
            for idx, record in unlabeled_df.iterrows()
        }

        # Select the next batch of examples based on uncertainty
        current_iteration_budget = min(iteration_budget, total_budget - used_budget)
        selected_indices = sorted(uncertainties.items(), key=operator.itemgetter(1), reverse=True)[:current_iteration_budget]
        next_batch_indices = [idx for idx, _ in selected_indices]

        # Add the selected examples to the training set and remove them from the unlabeled set
        new_training_df = unlabeled_df.loc[next_batch_indices]
        unlabeled_df = unlabeled_df.drop(next_batch_indices)
        training_df = pd.concat([training_df, new_training_df])

        # Update the used budget
        used_budget = training_df.shape[0]
        print(f"Elapsed time: {time.time() - start_time:.2f} seconds, Used budget: {used_budget}")

    # Train the final classifier on the full training set
    final_clf = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    X_train = training_df.iloc[:, 2:-1].apply(pd.to_numeric, errors='coerce')
    y_train = training_df['is_match']
    final_clf.fit(X_train, y_train)

    # Predict on the entire task dataset
    X_test = task_df.iloc[:, 2:-1].apply(pd.to_numeric, errors='coerce')
    task_df['pred'] = final_clf.predict(X_test)

    return task_df, used_budget


def active_learning_margin(task_df, iteration_budget, total_budget):
    """
    Apply active learning using margin sampling to label the records in the linkage task.

    Parameters:
    task_df (pd.DataFrame): DataFrame containing the task data.
    iteration_budget (int): The budget for each iteration.
    total_budget (int): The total budget available.

    Returns:
    pd.DataFrame: The labeled DataFrame.
    int: The total budget used.
    """
    start_time = time.time()

    # Initial training data selection using a random sample
    seed_indices = np.random.choice(task_df.shape[0], iteration_budget, replace=False)
    training_df = task_df.iloc[seed_indices]

    # Ensure diverse initial training set by checking for label diversity
    while len(training_df['is_match'].unique()) <= 1:
        seed_indices = np.random.choice(task_df.shape[0], iteration_budget, replace=False)
        training_df = task_df.iloc[seed_indices]

    # Create the unlabeled dataset by removing the selected training data
    unlabeled_df = task_df.drop(seed_indices)
    used_budget = training_df.shape[0]

    # Continue the active learning process until the budget is exhausted
    while used_budget < total_budget:
        clf = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
        X_train = training_df.iloc[:, 2:-1].apply(pd.to_numeric, errors='coerce')
        y_train = training_df['is_match']
        clf.fit(X_train, y_train)

        # Calculate margin for each unlabeled example
        margins = {
            idx: np.abs(clf.predict_proba([record.iloc[2:-1]])[0][0] - clf.predict_proba([record.iloc[2:-1]])[0][1])
            for idx, record in unlabeled_df.iterrows()
        }

        # Select the next batch of examples based on smallest margins
        current_iteration_budget = min(iteration_budget, total_budget - used_budget)
        selected_indices = sorted(margins.items(), key=lambda x: x[1])[:current_iteration_budget]
        next_batch_indices = [idx for idx, _ in selected_indices]

        # Add the selected examples to the training set and remove them from the unlabeled set
        new_training_df = unlabeled_df.loc[next_batch_indices]
        unlabeled_df = unlabeled_df.drop(next_batch_indices)
        training_df = pd.concat([training_df, new_training_df])

        # Update the used budget
        used_budget = training_df.shape[0]
        print(f"Elapsed time: {time.time() - start_time:.2f} seconds, Used budget: {used_budget}")

    # Train the final classifier on the full training set
    final_clf = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    X_train = training_df.iloc[:, 2:-1].apply(pd.to_numeric, errors='coerce')
    y_train = training_df['is_match']
    final_clf.fit(X_train, y_train)

    # Predict on the entire task dataset
    X_test = task_df.iloc[:, 2:-1].apply(pd.to_numeric, errors='coerce')
    task_df['pred'] = final_clf.predict(X_test)

    return task_df, used_budget


def label_linkage_tasks(selected_tasks, tasks_dir, tasks_info_df, min_budget, total_budget, labeled_tasks_dir, active_learning_strategy, relevant_columns):
    """
    Allocate budgets to selected linkage tasks and apply active learning to label them.

    Parameters:
    selected_tasks (dict): A dictionary of selected tasks.
    tasks_dir (str): The directory containing the linkage tasks files.
    tasks_info_df (pd.DataFrame): DataFrame containing information about linkage tasks.
    min_budget (int): The minimum budget to allocate per task.
    total_budget (int): The total available budget for allocation.
    labeled_tasks_dir (str): The directory where the labeled tasks will be saved.
    active_learning_strategy (str): The active learning strategy to use ("bootstrapping", "margin").
    relevant_columns (list): List of columns to be used for active learning.

    Returns:
    None
    """
    # Clear the labeled tasks directory
    delete_files(labeled_tasks_dir)

    # Allocate budgets to linkage tasks
    allocated_budgets = allocate_active_learning_budget(selected_tasks, tasks_dir, tasks_info_df, min_budget, total_budget)

    # Map active learning strategies to functions
    strategy_map = {
        "bootstrapping": active_learning_bootstrap,
        "margin": active_learning_margin,
    }

    # Validate the active learning strategy
    if active_learning_strategy not in strategy_map:
        raise ValueError(f"Invalid active learning strategy: {active_learning_strategy}")

    # Process each linkage task
    for task, budget in allocated_budgets.items():
        task_path = os.path.join(tasks_dir, task)
        task_df = pd.read_csv(task_path)

        # Prepare the dataframe for prediction
        processed_df = prepare_dataframe_prediction(task_df, relevant_columns)
        print(f"Processing task: {task}, initial shape: {processed_df.shape[0]}")

        # Apply the selected active learning strategy
        active_learning_function = strategy_map[active_learning_strategy]
        if task != 'www.canon-europe.com_cammarkt.com.csv':
            labeled_df, used_budget = active_learning_function(processed_df, min_budget, budget)
            print(f"Task {task} labeled with a budget of {used_budget}")

            # Save the labeled task
            labeled_df.to_csv(os.path.join(labeled_tasks_dir, task), index=False)
