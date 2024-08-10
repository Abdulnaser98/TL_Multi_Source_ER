import networkx as nx
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community import asyn_lpa_communities
from networkx.algorithms.community.quality import modularity
import sys
sys.path.append('/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/')
from utils import *



def detect_communities_using_girvan_newman(graph):
    """
    Detect communities in the given graph using the Girvan-Newman algorithm.

    Parameters:
    graph (networkx.Graph): The graph on which to perform community detection.

    Returns:
    list: A list of communities, each community being a list of nodes.
    """
    # Apply the Girvan-Newman algorithm to find communities
    community_generator = girvan_newman(graph)

    # Get the first set of communities
    first_communities = next(community_generator)
    communities = [list(community) for community in first_communities]

    # Calculate and print the modularity
    modularity_value = modularity(graph, first_communities)

    # Print the number of communities and the modularity value
    print(f"Number of communities: {len(communities)}")
    print(f"Modularity: {modularity_value}")

    return communities




def detect_communities_using_label_propagation_clustering(graph):
    """
    Detect communities in the given graph using the Asynchronous Label Propagation algorithm
    and print the communities and modularity value.

    Parameters:
    graph (networkx.Graph): The graph on which to perform community detection.

    Returns:
    list: A list of communities, each community being a list of nodes.
    """
    # Detect communities using asynchronous label propagation
    communities = list(asyn_lpa_communities(graph))

    # Calculate and print modularity
    modularity_value = modularity(graph, communities)

    # Print the number of communities and the modularity value
    print(f"Number of communities: {len(communities)}")
    print(f"Modularity: {modularity_value}")

    return communities


def detect_communities(community_detection_algorithum,graph):

    if community_detection_algorithum == 'girvan_newman':
       communities = detect_communities_using_girvan_newman(graph)
       return communities

    elif community_detection_algorithum == 'label_propagation_clustering':
       communities = detect_communities_using_label_propagation_clustering(graph)
       return communities

