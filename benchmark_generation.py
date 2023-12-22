import networkx as nx
import numpy as np
import random

def rewire_to_enforce_mu(G, mu, N_communities, eps=0.1):
    """
    Rewire edges in the graph to enforce the desired mixing parameter (mu) for each node.

    Parameters:
    - G (networkx.Graph): The input graph with community assignments and mu values.
    - mu (float): The desired mixing parameter.
    - N_communities (int): The number of communities in the graph.
    - eps (float, optional): The tolerance for mu difference. Defaults to 0.1.

    Returns:
    None
    """
    rewire_in, rewire_out = {}, {}
    for i in range(N_communities):
        rewire_in[i] = []
        rewire_out[i] = []
    for node in G.nodes():
        community = G.nodes[node]['community']
        current_mu = G.nodes[node]['mu']
        # Identify nodes to rewire based on the difference between current mu and desired mu
        if np.abs(current_mu-mu) > eps and G.degree[node] > 1:
            if current_mu > mu:
                rewire_out[community] += [node]
            else:
                rewire_in[community] += [node]
    # Perform rewiring for in-community and out-community edges
    rewire_to_enforce_mu_in(G, mu, rewire_in, rewire_out, eps)
    rewire_to_enforce_mu_out(G, mu, rewire_out, eps)

def update_mu(G, nodes):
    """
    Update the mu values for a list of nodes in a graph.

    Parameters:
    - G (networkx.Graph): The input graph with community assignments and mu values.
    - nodes (list): List of nodes for which to update the mu values.

    Returns:
    None
    """
    for node in nodes:
        G.nodes[node]['mu'] = len(G.nodes[node]['in_community_neighbors']) / G.degree[node]

def update_rewire_list_for_node(G, node, neighbor_community, mu, eps, rewire_in, rewire_out):
    """
    Update rewire lists for a specific node based on its mu value.

    Parameters:
    - node_community (int): Community of the node.
    - node (int): Node whose mu value is checked for rewire lists.
    - mu (float): The desired mixing parameter.
    - eps (float): The tolerance for mu difference.
    - rewire_in (dict): Dictionary containing nodes to rewire for each community.
    - rewire_out (dict): Dictionary containing nodes to rewire for each community (out-community edges).

    Returns:
    None
    """
    try:
        rewire_in[G.nodes[node]['community']].remove(node)
    except:
        pass
    # Add node to rewire list based on mu values
    if np.abs(G.nodes[node]['mu'] - mu) > eps:
        if G.nodes[node]['mu'] - mu > eps and node not in rewire_out[neighbor_community]:
            rewire_out[G.nodes[node]['community']].append(node)
        elif G.nodes[node]['mu'] - mu < eps and node not in rewire_in[neighbor_community]:
            rewire_in[G.nodes[node]['community']].append(node)

def update_rewire_list(G, neighbor1, neighbor2, mu, eps, rewire_in, rewire_out):
    """
    Update rewire lists based on the mu values of neighbors.

    Parameters:
    - G (networkx.Graph): The input graph with community assignments and mu values.
    - neighbor1, neighbor2 (int): Nodes whose mu values are checked for rewire lists.
    - mu (float): The desired mixing parameter.
    - eps (float): The tolerance for mu difference.
    - rewire_in (dict): Dictionary containing nodes to rewire for each community.
    - rewire_out (dict): Dictionary containing nodes to rewire for each community (out-community edges).

    Returns:
    None
    """
    update_rewire_list_for_node(G, neighbor1, G.nodes[neighbor2]['community'], mu, eps, rewire_in, rewire_out)
    update_rewire_list_for_node(G, neighbor2, G.nodes[neighbor2]['community'], mu, eps, rewire_in, rewire_out)

def rewire_nodes_in(G, node1, neighbor1, node2, neighbor2, mu, eps, rewire_in, rewire_out, to_rewire):
    """
    Rewire edges in the graph and update mu values based on the desired mixing parameter.

    Parameters:
    - G (networkx.Graph): The input graph with community assignments and mu values.
    - node1, neighbor1, node2, neighbor2 (int): Nodes and neighbors to be rewired.
    - mu (float): The desired mixing parameter.
    - eps (float): The tolerance for mu difference.
    - rewire_in (dict): Dictionary containing nodes to rewire for each community.
    - rewire_out (dict): Dictionary containing nodes to rewire for each community (out-community edges).
    - to_rewire (list): List of nodes to be rewired.

    Returns:
    None
    """
    # Remove existing edges
    G.remove_edge(node1, neighbor1)
    G.remove_edge(node2, neighbor2)
    # Add new edges for rewiring   
    G.add_edge(node1, node2)
    G.add_edge(neighbor1, neighbor2)
    # Update neighbors
    G.nodes[node1]['out_community_neighbors'].remove(neighbor1)
    G.nodes[node2]['out_community_neighbors'].remove(neighbor2)
    G.nodes[neighbor1]['out_community_neighbors'].remove(node1)
    G.nodes[neighbor2]['out_community_neighbors'].remove(node2)
    G.nodes[node1]['in_community_neighbors'].append(node2)
    G.nodes[node2]['in_community_neighbors'].append(node1)
    if G.nodes[neighbor1]['community'] == G.nodes[neighbor2]['community']:
        G.nodes[neighbor1]['in_community_neighbors'].append(neighbor2)
        G.nodes[neighbor2]['in_community_neighbors'].append(neighbor1)
    else:
        G.nodes[neighbor1]['out_community_neighbors'].append(neighbor2)
        G.nodes[neighbor2]['out_community_neighbors'].append(neighbor1)
    # Update mu values
    update_mu(G, [node1, node2, neighbor1, neighbor2])
    # Remove nodes from to_rewire if their mu values are within tolerance
    if G.nodes[node1]['mu'] - mu >= -eps:
        to_rewire.remove(node1)
    if G.nodes[node2]['mu'] - mu >= -eps:
        to_rewire.remove(node2)
    # Update rewire lists for neighbors
    update_rewire_list(G, neighbor1, neighbor2, mu, eps, rewire_in, rewire_out)

def rewire_nodes_out(G, node1, neighbor1, node2, neighbor2, mu, eps, rewire_out, to_rewire, rewire_with):
    """
    Rewire edges in a graph while updating community structure and mu values.

    Parameters:
    - G (networkx.Graph): The input graph.
    - node1, neighbor1, node2, neighbor2: Nodes and neighbors involved in the rewiring.
    - mu (float): Target mu value.
    - eps (float): Tolerance for mu values.
    - rewire_out (dict): Dictionary to store nodes to rewire based on community.
    - to_rewire (list): List of nodes to potentially rewire.
    - rewire_with (list): List of nodes to rewire with.
    
    Returns:
    None
    """
    # Remove existing edges
    G.remove_edge(node1, neighbor1)
    G.remove_edge(node2, neighbor2)
    # Add new edges for rewiring   
    G.add_edge(node1, node2)
    G.add_edge(neighbor1, neighbor2)
    # Update neighbors
    G.nodes[node1]['in_community_neighbors'].remove(neighbor1)
    G.nodes[node2]['in_community_neighbors'].remove(neighbor2)
    G.nodes[neighbor1]['in_community_neighbors'].remove(node1)
    G.nodes[neighbor2]['in_community_neighbors'].remove(node2)
    G.nodes[node1]['out_community_neighbors'].append(node2)
    G.nodes[node2]['out_community_neighbors'].append(node1)
    if G.nodes[neighbor1]['community'] == G.nodes[neighbor2]['community']:
        G.nodes[neighbor1]['in_community_neighbors'].append(neighbor2)
        G.nodes[neighbor2]['in_community_neighbors'].append(neighbor1)
    else:
        G.nodes[neighbor1]['out_community_neighbors'].append(neighbor2)
        G.nodes[neighbor2]['out_community_neighbors'].append(neighbor1)
    # Update mu values
    update_mu(G, [node1, node2, neighbor1, neighbor2])
    # Remove nodes if their mu values are within tolerance
    for n in [node1, node2, neighbor1, neighbor2]:
        if G.nodes[n]['mu'] - mu <= eps:
            if n in to_rewire:
                to_rewire.remove(n)
            elif n in rewire_with:
                rewire_with.remove(n)
                rewire_out[G.nodes[n]['community']].remove(n)

def rewire_to_enforce_mu_in(G, mu, rewire_in, rewire_out, eps):
    """
    Rewire edges in the graph to enforce the desired mixing parameter (mu) for in-community nodes.

    Parameters:
    - G (networkx.Graph): The input graph with community assignments and mu values.
    - mu (float): The desired mixing parameter.
    - rewire_in (dict): Dictionary containing nodes to rewire for each community.
    - rewire_out (dict): Dictionary containing nodes to rewire for each community (out-community edges).
    - eps (float): The tolerance for mu difference.

    Returns:
    None
    """
    for community in rewire_in:
        to_rewire = rewire_in[community]
        N = len(to_rewire)
        i = 0
        # Iterate until no more than one node remains for rewiring or the maximum attempts are reached
        while len(to_rewire) > 1 and i < N*G.number_of_nodes():
            i += 1
            node1, node2 = random.sample(to_rewire, 2)
            neighbor1 = random.choice(G.nodes[node1]['out_community_neighbors'])
            neighbor2 = random.choice(G.nodes[node2]['out_community_neighbors'])
            j = 0
            # Ensure the selected nodes don't create loops
            while (node1 == node2 or node2 in G.nodes[node1]['in_community_neighbors'] or neighbor1 == neighbor2 or neighbor2 in list(G[neighbor1].copy().keys())) and j < G.number_of_nodes():
                j += 1
                node2 = random.choice(to_rewire)
                neighbor2 = random.choice(G.nodes[node2]['out_community_neighbors'])
            # If the maximum attempts are reached, we consider it impossible to rewire node1
            if j == G.number_of_nodes():
                to_rewire.remove(node1)
            else:
                # Rewire the selected nodes and neighbors
                rewire_nodes_in(G, node1, neighbor1, node2, neighbor2, mu, eps, rewire_in, rewire_out, to_rewire)

def rewire_to_enforce_mu_out(G, mu, rewire_out, eps):
    """
    Rewire edges in the graph to enforce the desired mixing parameter (mu) for out-community nodes.

    Parameters:
    - G (networkx.Graph): The input graph with community assignments and mu values.
    - mu (float): The desired mixing parameter.
    - rewire_out (dict): Dictionary containing nodes to rewire for each community (out-community edges).
    - eps (float): The tolerance for mu difference.

    Returns:
    None
    """
    for community in rewire_out:
        to_rewire = rewire_out[community]
        rewire_with = [rewire_out[c] for c in rewire_out if c != community]
        rewire_with = [rr for r in rewire_with for rr in r]
        N = len(to_rewire)
        i = 0
        # Iterate until no more than one node remains for rewiring or the maximum attempts are reached
        while len(to_rewire) > 1 and i < N*G.number_of_nodes():
            i += 1
            node1 = random.choice(to_rewire)
            node2 = random.choice(rewire_with)
            neighbor1 = random.choice(G.nodes[node1]['in_community_neighbors'])
            neighbor2 = random.choice(G.nodes[node2]['in_community_neighbors'])
            j = 0
            # Ensure the selected nodes don't create loops or undo the work done in the previous function
            while (neighbor1 == neighbor2 or neighbor2 in list(G[neighbor1].copy().keys()) or (neighbor2 not in rewire_with and node2 not in to_rewire) or neighbor1 not in to_rewire or G.nodes[neighbor2]['community'] == community) and j < G.number_of_nodes():
                j += 1
                node2 = random.choice(rewire_with)
                neighbor1 = random.choice(G.nodes[node1]['in_community_neighbors'])
                neighbor2 = random.choice(G.nodes[node2]['in_community_neighbors'])
            # If the maximum attempts are reached, we consider it impossible to rewire node1
            if j == G.number_of_nodes():
                to_rewire.remove(node1)
            else:
                # Rewire the selected nodes and neighbors
                rewire_nodes_out(G, node1, neighbor1, node2, neighbor2, mu, eps, rewire_out, to_rewire, rewire_with)

def compute_mu(G):
    """
    Compute the mixing parameter (mu) for each node in the graph.

    Parameters:
    - G (networkx.Graph): The input graph with community assignments.

    Returns:
    None
    """
    for node in G.nodes():
        community = G.nodes[node]['community']
        neighbors = list(G.neighbors(node))
        in_community_neighbors = [neighbor for neighbor in neighbors if G.nodes[neighbor]['community'] == community]
        out_community_neighbors = [neighbor for neighbor in neighbors if G.nodes[neighbor]['community'] != community]
        mu = len(in_community_neighbors) / len(neighbors) if len(neighbors) > 0 else 0
        G.nodes[node]['mu'] = mu
        G.nodes[node]['in_community_neighbors'] = in_community_neighbors
        G.nodes[node]['out_community_neighbors'] = out_community_neighbors

def assign_nodes_to_communities(G, community_sizes):
    """
    Assign nodes in a graph to communities based on specified community sizes.

    Parameters:
    - G (networkx.Graph): The input graph.
    - community_sizes (list): A list of integers representing the desired sizes of communities.

    Returns:
    None
    """
    nodes = list(G.nodes())
    N_communities = len(community_sizes)
    current_community_sizes = [0]*N_communities
    for node in nodes:
        G.nodes[node]['community'] = -1
    while nodes:
        np.random.shuffle(nodes)
        to_remove = []
        to_add = []
        for i, node in enumerate(nodes):
            community = np.random.randint(N_communities)
            # If it has only one neighbor, assign the node to the same community as its neighbor
            if G.degree[node] == 1:
                neighbor = list(G[node].copy().keys())[0]
                if G.nodes[neighbor]['community'] != -1:
                    G.nodes[node]['community'] = G.nodes[neighbor]['community']
                    to_remove += [node]
            elif community_sizes[community] > len([neighbor for neighbor in G.neighbors(node) if G.nodes[neighbor]['community'] == community]):
                G.nodes[node]['community'] = community
                to_remove += [node]
                current_community_sizes[community] += 1
                if current_community_sizes[community] > community_sizes[community]:
                    to_add += [random.choice([node_in_community for node_in_community, data in G.nodes(data=True) if data.get('community') == community])]
                    G.nodes[to_add[-1]]['community'] = -1


        nodes = [node for node in nodes if node not in to_remove]
        nodes += to_add

def generate_benchmark(N, hki, gamma, beta, mu, hki_community = None, save_path=None):
    """
    Generate a benchmark graph with specified parameters using a configuration model.

    Parameters:
    - N (int): Number of nodes in the graph.
    - hki (float): Power-law exponent for generating node degrees.
    - gamma (float): Power-law exponent for generating community sizes.
    - beta (float): Power-law exponent for generating community sizes.
    - mu (float): Mixing parameter to control inter-community edges.
    - hki_community (int, optional): Maximum node degree within communities. Defaults to 500.

    Returns:
    - networkx.Graph: A graph generated based on the specified parameters.
    """
    # Step 1: Generate node degrees from a power-law distribution
    degrees = np.random.power(gamma, N)
    mean = np.mean(degrees)
    rounding = hki/mean
    degrees = np.round(degrees * (rounding+1)).astype(int)
    k_max = np.max(degrees)
    k_min = np.min(degrees)
    if not hki_community:
        hki_community = random.randint(k_max, int(N/2))
    # Step 2: Create a configuration model to connect nodes
    try:
        G = nx.configuration_model(degrees)
    except:
        # Adjust the last degree to avoid issues with configuration_model
        degrees[-1] += 1
        G = nx.configuration_model(degrees)
    # Create an undirected graph from the configuration model
    G = nx.Graph(G)
    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))
    # Step 3: Generate community sizes from a power-law distribution
    s_min, s_max = 0, 0
    while s_min <= k_min or s_max <= k_max:
        community_sizes = np.random.power(beta, N)
        community_sizes = np.round(community_sizes * (hki_community + 1)).astype(int)
        # Ensure the sum of community sizes equals the total number of nodes
        community_sizes = community_sizes[community_sizes.cumsum() <= N].tolist()
        community_sizes += [N - np.sum(community_sizes)]
        s_min = np.min(community_sizes)
        s_max = np.max(community_sizes)

    # Step 4: Assign nodes to communities
    assign_nodes_to_communities(G, community_sizes)

    compute_mu(G)

    # Step 5: Rewire to enforce the mixing parameter mu
    rewire_to_enforce_mu(G, mu, len(community_sizes))

    # Store the generated graph
    if save_path:
        nx.write_gml(G, save_path+".gml")
    return G