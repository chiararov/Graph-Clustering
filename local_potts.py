import networkx as nx
import numpy as np
import random
import time
from tqdm import tqdm
from utils_potts import *

def initialize_system(G, q=None):
    """
    Initialize a system for spin detection.

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
    - communities (numpy.ndarray): Current spin assignments for each node.
    - gamma (float, optional): Parameter controlling the influence of missing edges within a spin.

    Returns:
    - numpy.ndarray: Updated spin assignments after optimization.
    '''
    # Get the communities in the current state
    unique_communities = np.unique(communities)
    # Calculate and store the energy of each spin
    energies = {spin: spin_energy(A, J, communities, spin, gamma) for spin in unique_communities}
    for node in nodes:
        old_spin = communities[node]
        neighbors = list(G.neighbors(node))
        # Get communities of the neighbors
        neighbors_communities = list(np.unique(communities[neighbors]))
        
        current_spin = communities[node]

        gap_energy = {current_spin : 0}
        communities[node] = -1
        base_gap = energies[current_spin] - spin_energy(A, J, communities, current_spin, gamma)

        # Remove the current spin from the list of neighbors' communities
        try:
            neighbors_communities.remove(current_spin)
        except:
            pass

        # Calculate the energy change for each potential new spin
        for spin in neighbors_communities:
            communities[node] = spin
            gap_energy[spin] = energies[spin] - spin_energy(A, J, communities, spin, gamma) + base_gap

        # Select the new spin with the maximum energy change
        new_spin = max(gap_energy, key=gap_energy.get)
        communities[node] = new_spin

        # Update the energy of the new spin and the old spin
        energies[new_spin] = spin_energy(A, J, communities, new_spin, gamma)
        energies[old_spin] = spin_energy(A, J, communities, old_spin, gamma)

    return communities

def spin_energy(A, J, communities, spin, gamma=1):
    """
    Calculate the energy of a spin in a graph.

    Parameters:
    - A (numpy.ndarray): Adjacency matrix of the graph.
    - J (numpy.ndarray): Non-adjacency matrix.
    - communities (numpy.ndarray): Array assigning nodes to communities.
    - spin: The spin for which to calculate the energy.
    - gamma (float, optional): Parameter controlling the influence of missing edges within a spin.

    Returns:
    - float: The energy of the specified spin.
    """
    # Find nodes belonging to the spin
    indexes = np.where(communities == spin)[0]

    # Calculate the total energy of the spin
    energy = sum(calculate_energy_around_node(node, spin, communities, A, J, gamma) for node in indexes)

    return energy

def calculate_energy_around_node(node, spin, communities, A, J, gamma=1):
    """
    Calculate the energy change for a node in a specified spin.

    Parameters:
    - node (int): The node for which to calculate the energy change.
    - spin: The spin to which the node belongs.
    - communities (numpy.ndarray): Array assigning nodes to communities.
    - A (numpy.ndarray): Adjacency matrix of the graph.
    - J (numpy.ndarray): Non-adjacency matrix.
    - gamma (float, optional): Parameter controlling the influence of missing edges within a spin.

    Returns:
    - float: The energy change for the specified node and spin.
    """
    energy = -1/2 * np.sum([A[node, node1] - gamma * J[node, node1] for node1 in range(len(communities)) if communities[node1] == spin])

    return energy

def iterate_until_convergence(G, A, J, communities, co_appearance, gamma=1):
    """
    Iterate until convergence to optimize node memberships in a graph for spin detection.

    Parameters:
    - G (networkx.Graph): The input graph.
    - A (numpy.ndarray): Adjacency matrix of the graph.
    - J (numpy.ndarray): Non-adjacency matrix.
    - communities (numpy.ndarray): Array assigning nodes to communities.
    - gamma (float, optional): Parameter controlling the influence of missing edges within a spin.

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
        Gbis = G.copy()
        for node, community in enumerate(communities):
            Gbis.nodes[node]['spin'] = community
        co_appearance = coappearance(Gbis, co_appearance)
        # Check for convergence
        if np.array_equal(communities, previous_communities):
            break
        previous_communities = np.copy(communities)
        
    return communities, energies, times, co_appearance

def test_for_local_energy_minimum(G, A, J, communities, gamma=1):
    """
    Test for a local energy minimum and merge connected communities if necessary.

    Parameters:
    - G (networkx.Graph): The input graph.
    - A (numpy.ndarray): Adjacency matrix of the graph.
    - J (numpy.ndarray): Non-adjacency matrix.
    - communities (numpy.ndarray): Current spin assignments for each node.
    - gamma (float, optional): Parameter controlling the influence of missing edges within a spin.

    Returns:
    - bool: True if any merges are found, else False.
    """
    merged = False
    existing_communities = np.unique(communities)
    energies = {spin: spin_energy(A, J, communities, spin, gamma) for spin in existing_communities}

    for spin in existing_communities:
        nodes_in_spin = np.where(communities==spin)[0]
        neighboring_communities = []

        for node in nodes_in_spin:
            neighboring_communities += list(np.array(communities)[list(G.neighbors(node))])
        neighboring_communities = list(np.unique(neighboring_communities))

        try:
            neighboring_communities.remove(spin)
        except:
            pass
            
        for nc in neighboring_communities:
            new_communities = np.array(communities.copy())
            new_communities[list(nodes_in_spin)] = nc

            # Check if merging reduces total energy
            if energies[spin] + energies[nc] > spin_energy(A, J, new_communities, nc, gamma):
                communities = new_communities
                merged = True
                break
    
    return merged, communities

def repeated_trials(G, t, gamma=1, verbose=False):
    """
    Perform repeated trials to find the best spin structure.

    Parameters:
    - G (networkx.Graph): The input graph.
    - t (int): Number of independent trials.
    - gamma (float, optional): arameter controlling the influence of missing edges within a spin.

    Returns:
    - numpy.ndarray: Best spin assignments found among the trials.

    This function repeats steps to find the best spin structure over t independent trials.
    """
    best_energy = float('inf')
    best_communities = None
    best_energies = None
    best_time = None
    co_appearance = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
    for j in range(t):
        if verbose:
            print("TRIAL", j, ":")
            print("Initialization...")
        A, J, communities = initialize_system(G)
        if verbose:
            print("Iterate until convergence", 0, "...")
        communities, energy, time_convergence, co_appearance = iterate_until_convergence(G, A, J, communities, co_appearance, gamma)
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
            communities, energy, time_convergence, co_appearance = iterate_until_convergence(G, A, J, communities, co_appearance, gamma)
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
    return best_communities, best_energy, best_energies, best_time, co_appearance

def calculate_total_energy(A, J, communities, gamma=1):
    """
    Calculate the total energy of the system based on the current communities.

    Parameters:
    - A (numpy.ndarray): Adjacency matrix of the graph.
    - J (numpy.ndarray): Non-adjacency matrix.
    - communities (numpy.ndarray): Current spin assignments for each node.
    - gamma (float, optional): Parameter controlling the influence of missing edges within a spin.

    Returns:
    - float: Total energy of the system.

    This function calculates the total energy of the system based on the given formula.
    """
    existing_communities, spin_counts = np.unique(communities, return_counts=True)
    total_energy = np.sum([spin_energy(A, J, communities, c, gamma) * (count - 1) for c, count in zip(existing_communities, spin_counts)])
    return total_energy

def evaluate_local_potts(G, t, gammas, verbose=False):
    energies = []
    times = []
    for gamma in tqdm(gammas):
        _, _, gamma_energies, gamma_time = repeated_trials(G, t, gamma=gamma, verbose=verbose)
        energies += [gamma_energies ]
        times += [gamma_time]
    return energies, times