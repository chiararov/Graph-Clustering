import networkx as nx
import numpy as np
import random
import time

def initialize_system(G, q=None):
    """
    Initialize a system for community detection.

    Parameters:
    - G (networkx.Graph): The input graph.
    - q (int, optional): The number of communities. If not provided, defaults to the number of nodes in the graph.

    Returns:
    - numpy.ndarray: Adjacency matrix of the graph.
    - numpy.ndarray: Non-adjacency matrix
    - numpy.ndarray: Array assigning nodes to communities.
    """
    # Get the adjacency matrix
    A = nx.adjacency_matrix(G, weight=None).toarray()

    # Create the non-adjacency matrix J with null diagonal
    J = 1 - A
    np.fill_diagonal(J, 0)

    # Set default value for q if not provided
    if q is None:
        q = G.number_of_nodes()

    # Assign nodes to communities
    communities = np.array([node % q for node in range(G.number_of_nodes())])

    return A, J, communities

def optimize_node_memberships(G, A, J, communities, nodes, gamma=1):
    '''
    Optimize node memberships in a network based on energy minimization.

    Parameters:
    - G (networkx.Graph): The input graph.
    - A (numpy.ndarray): Adjacency matrix of the graph.
    - J (numpy.ndarray): Non-adjacency matrix.
    - communities (numpy.ndarray): Current community assignments for each node.
    - gamma (float, optional): Parameter controlling the influence of missing edges within a community.

    Returns:
    - numpy.ndarray: Updated community assignments after optimization.
    '''
    # Get the communities in the current state
    unique_communities = np.unique(communities)
    # Calculate and store the energy of each community
    energies = {community: community_energy(A, J, communities, community, gamma) for community in unique_communities}
    for node in nodes:
        old_community = communities[node]
        neighbors = list(G.neighbors(node))
        # Get communities of the neighbors
        neighbors_communities = list(np.unique(communities[neighbors]))
        
        current_community = communities[node]

        gap_energy = {current_community : 0}
        communities[node] = -1
        base_gap = energies[current_community] - community_energy(A, J, communities, current_community, gamma)

        # Remove the current community from the list of neighbors' communities
        try:
            neighbors_communities.remove(current_community)
        except:
            pass

        # Calculate the energy change for each potential new community
        for community in neighbors_communities:
            communities[node] = community
            gap_energy[community] = energies[community] - community_energy(A, J, communities, community, gamma) + base_gap

        # Select the new community with the maximum energy change
        new_community = max(gap_energy, key=gap_energy.get)
        communities[node] = new_community

        # Update the energy of the new community and the old community
        energies[new_community] = community_energy(A, J, communities, new_community, gamma)
        energies[old_community] = community_energy(A, J, communities, old_community, gamma)

    return communities

def community_energy(A, J, communities, community, gamma=1):
    """
    Calculate the energy of a community in a graph.

    Parameters:
    - A (numpy.ndarray): Adjacency matrix of the graph.
    - J (numpy.ndarray): Non-adjacency matrix.
    - communities (numpy.ndarray): Array assigning nodes to communities.
    - community: The community for which to calculate the energy.
    - gamma (float, optional): Parameter controlling the influence of missing edges within a community.

    Returns:
    - float: The energy of the specified community.
    """
    # Find nodes belonging to the community
    indexes = np.where(communities == community)[0]

    # Calculate the total energy of the community
    energy = sum(calculate_energy_around_node(node, community, communities, A, J, gamma) for node in indexes)

    return energy

def calculate_energy_around_node(node, community, communities, A, J, gamma=1):
    """
    Calculate the energy change for a node in a specified community.

    Parameters:
    - node (int): The node for which to calculate the energy change.
    - community: The community to which the node belongs.
    - communities (numpy.ndarray): Array assigning nodes to communities.
    - A (numpy.ndarray): Adjacency matrix of the graph.
    - J (numpy.ndarray): Non-adjacency matrix.
    - gamma (float, optional): Parameter controlling the influence of missing edges within a community.

    Returns:
    - float: The energy change for the specified node and community.
    """
    energy = -1/2 * np.sum([A[node, node1] - gamma * J[node, node1] for node1 in range(len(communities)) if communities[node1] == community])

    return energy

def iterate_until_convergence(G, A, J, communities, gamma=1):
    """
    Iterate until convergence to optimize node memberships in a graph for community detection.

    Parameters:
    - G (networkx.Graph): The input graph.
    - A (numpy.ndarray): Adjacency matrix of the graph.
    - J (numpy.ndarray): Non-adjacency matrix.
    - communities (numpy.ndarray): Array assigning nodes to communities.
    - gamma (float, optional): Parameter controlling the influence of missing edges within a community.

    Returns:
    - numpy.ndarray: Updated array assigning nodes to communities.
    """
    previous_communities = np.copy(communities)
    nodes = list(np.arange(len(communities)))
    random.shuffle(nodes)
    energies = []
    times = []
    while True:
        # Optimize node memberships
        start_time = time.time()
        communities = optimize_node_memberships(G, A, J, communities, nodes, gamma)
        times += [time.time() - start_time]
        energies += [calculate_total_energy(A, J, communities, gamma)]
        # Check for convergence
        if np.array_equal(communities, previous_communities):
            break
        previous_communities = np.copy(communities)

    return communities, energies, times

def test_for_local_energy_minimum(G, A, J, communities, gamma=1):
    """
    Test for a local energy minimum and merge connected communities if necessary.

    Parameters:
    - G (networkx.Graph): The input graph.
    - A (numpy.ndarray): Adjacency matrix of the graph.
    - J (numpy.ndarray): Non-adjacency matrix.
    - communities (numpy.ndarray): Current community assignments for each node.
    - gamma (float, optional): Parameter controlling the influence of missing edges within a community.

    Returns:
    - bool: True if any merges are found, else False.
    """
    merged = False
    existing_communities = np.unique(communities)
    energies = {community: community_energy(A, J, communities, community, gamma) for community in existing_communities}

    for community in existing_communities:
        nodes_in_community = np.where(communities==community)[0]
        neighboring_communities = []

        for node in nodes_in_community:
            neighboring_communities += list(np.array(communities)[list(G.neighbors(node))])
        neighboring_communities = list(np.unique(neighboring_communities))

        try:
            neighboring_communities.remove(community)
        except:
            pass
            
        for nc in neighboring_communities:
            new_communities = np.array(communities.copy())
            new_communities[list(nodes_in_community)] = nc

            # Check if merging reduces total energy
            if energies[community] + energies[nc] > community_energy(A, J, new_communities, nc, gamma):
                communities = new_communities
                merged = True
                break
    
    return merged, communities

def repeated_trials(G, t, gamma=1, verbose=False):
    """
    Perform repeated trials to find the best community structure.

    Parameters:
    - G (networkx.Graph): The input graph.
    - t (int): Number of independent trials.
    - gamma (float, optional): arameter controlling the influence of missing edges within a community.

    Returns:
    - numpy.ndarray: Best community assignments found among the trials.

    This function repeats steps to find the best community structure over t independent trials.
    """
    best_energy = float('inf')
    best_communities = None
    best_energies = None
    best_time = None
    for j in range(t):
        if verbose:
            print("TRIAL", j, ":")
            print("Initialization...")
        A, J, communities = initialize_system(G)
        if verbose:
            print("Iterate until convergence", 0, "...")
        communities, energy, time_convergence = iterate_until_convergence(G, A, J, communities, gamma)
        times = time_convergence
        e = energy
        i = 0
        if verbose:
            print("Test for a local energy minimum...")
        merged, communities = test_for_local_energy_minimum(G, A, J, communities, gamma)
        while merged:
            i += 1
            if verbose:
                print("Iterate until convergence", i, "...")
            communities, energy, time_convergence = iterate_until_convergence(G, A, J, communities, gamma)
            times += time_convergence
            e += energy
            if verbose:
                print("Test for a local energy minimum...")
            merged, communities = test_for_local_energy_minimum(G, A, J, communities, gamma)

        # Calculate the energy of the final solution
        energy = calculate_total_energy(A, J, communities, gamma)

        # Update the best solution if the current energy is lower
        if energy < best_energy:
            best_energy = energy
            best_energies = e
            best_communities = np.copy(communities)
            best_time = times
    return best_communities, best_energy, best_energies, best_time

def calculate_total_energy(A, J, communities, gamma=1):
    """
    Calculate the total energy of the system based on the current communities.

    Parameters:
    - A (numpy.ndarray): Adjacency matrix of the graph.
    - J (numpy.ndarray): Non-adjacency matrix.
    - communities (numpy.ndarray): Current community assignments for each node.
    - gamma (float, optional): Parameter controlling the influence of missing edges within a community.

    Returns:
    - float: Total energy of the system.

    This function calculates the total energy of the system based on the given formula.
    """
    existing_communities, community_counts = np.unique(communities, return_counts=True)
    total_energy = np.sum([community_energy(A, J, communities, c, gamma) * (count - 1) for c, count in zip(existing_communities, community_counts)])
    return total_energy