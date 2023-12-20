"""
Graph Clustering - MVA - Dec 2023
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def generate_known_cluster(N_nodes, N_clusters, k_in, k_out):
    
    network = nx.Graph()
    nodes_per_cluster = N_nodes//N_clusters
    for i in range(N_nodes):
        cluster = i//nodes_per_cluster
        network.add_node(i)
        network.nodes[i]["spin"] = cluster 
    
    
    #Intra-cluster connectivity:
    for i in range(N_nodes):
        cluster = i//nodes_per_cluster
        selected_nodes = [n for n,v in network.nodes(data=True) if v['spin'] == cluster]
        for node in selected_nodes:
            if (node!=i) and (node,i) not in network.edges() and np.random.rand()<k_in:
                network.add_edge(node,i)
        
        for _ in range(nodes_per_cluster):
            random_cluster = np.random.randint(N_clusters)
            while random_cluster == cluster:
                random_cluster = np.random.randint(N_clusters)
            selected_random_cluster = [n for n,v in network.nodes(data=True) if v['spin'] == random_cluster]
            ind_random_node = np.random.randint(len(selected_random_cluster))
            selected_node = selected_random_cluster[ind_random_node]
                
            if (i,selected_node) not in network.edges() and np.random.rand()<k_out:
                network.add_edge(selected_node,i)
                
        
    return network

def plot_graph_cluster(G,title):
    plt.figure(figsize=(10,6))
    pos = nx.spring_layout(G)
    node_states = nx.get_node_attributes(G, 'spin')
    state_pos = {n: (x, y) for n, (x,y) in pos.items()}
    nx.draw(G,pos)
    nx.draw_networkx_labels(G, state_pos, labels=node_states, font_color='red')
    plt.title(title)
    plt.show()

def quality(true, pred):
    TP = 0
    FN = 0
    FP=0
    TN=0 
    for i in range(len(true)):
        for j in range(len(pred)):
            if true[i] ==true[j] and pred[i] ==pred[j]:
                TP+=1
            elif true[i] == true[j] and pred[i] != pred[j]:
                FN+=1
            elif true[i] != true[j] and pred[i] == pred[j]:
                FP+=1
            if true[i] != true[j] and pred[i] != pred[j]:
                TN+=1
    if TP+FN==0:
        sens=0
    else:
        sens=TP/(TP+FN)
    if TN+FP==0:
        spe=0
    else:
        spe=TN/(TN+FP)
    return sens,spe
