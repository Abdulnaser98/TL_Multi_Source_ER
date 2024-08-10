import pandas as pd
import sys
sys.path.append('/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/')
from utils import *

def select_largest_file_in_community(path_to_sim_vector_folder,community, node_mapping):
    """
    Select the largest file in each community and return the largest file and a list of other linkage tasks.

    Parameters:
    community (list): A list of nodes representing the community.
    node_mapping (dict): A dictionary mapping node labels to community nodes.

    Returns:
    tuple: A tuple containing the largest file and a list of other linkage tasks.
    """
    # Identify linkage tasks that belong to the community
    community_tasks = [task for task, node in node_mapping.items() if node in community]

    # Find the task with the maximum file count
    largest_task = None
    largest_task_count = -1
    for task in community_tasks:
        task_count = get_sim_vec_file_length(path_to_sim_vector_folder,task)
        if task_count > largest_task_count:
            largest_task = task
            largest_task_count = task_count

    # Create a list of other tasks in the community
    other_tasks = [task for task in community_tasks if task != largest_task]

    return largest_task, other_tasks


def select_linkage_tasks_from_communities(path_to_sim_vector_folder,linkage_tasks_communities, node_mapping):
    """
    Loop over each community to select the largest file and related linkage tasks,
    then store the results in a dictionary.

    Parameters:
    linkage_tasks_communities (list): A list of communities, each community being a list of nodes.
    node_mapping (dict): A dictionary mapping node labels to community nodes.

    Returns:
    dict: A dictionary where keys are the largest file and values are lists of other linkage tasks.
    """
    selected_tasks_dict = {}

    for community in linkage_tasks_communities:
        largest_task, other_tasks = select_largest_file_in_community(path_to_sim_vector_folder,community, node_mapping)
        selected_tasks_dict[largest_task] = other_tasks

    return selected_tasks_dict

