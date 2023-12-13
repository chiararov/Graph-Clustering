import itertools
import matplotlib.pyplot as plt
import networkx as nx
import random
import math
from Utils import modularity
import numpy as np
import scipy.sparse as sp

def simulated_annealing(graph, initial_partition, temperature, cooling_rate, iterations):
    current_partition = initial_partition.copy()
    current_score = modularity(graph, current_partition)

    best_partition = current_partition.copy()
    best_score = current_score
    scores=[]

    for _ in range(iterations):
        new_partition = perturb_partition(current_partition)
        new_score = modularity(graph, new_partition)

        
        if new_score > current_score:
            current_partition = new_partition
            current_score = new_score

        #pour eviter le depassement numerique, on change la condition classique
        proba=math.log(random.random() + 1e-10 )
        comparaison=(new_score - current_score) / temperature
        
        if proba < comparaison:
            best_partition = new_partition
            best_score = new_score

        temperature *= 1 - cooling_rate
        
        scores.append(best_score)

    return best_partition, best_score,scores

def perturb_partition(partition):
    perturbed_partition = partition.copy()
    node = random.choice(list(partition.keys()))
    
    # Vérifier si l'ensemble de communautés est vide
    available_communities = set(partition.values()) - {partition[node]}
    if not available_communities:
        new_community = random.choice(list(set(partition.values())))
    else:
        new_community = random.choice(list(available_communities))
    
    perturbed_partition[node] = new_community
    return perturbed_partition


def grid_search(graph, temperature_range, cooling_rate_range, iterations, k_range):
    best_score = -float('inf')
    best_params = {}
    print("Debug in grid_search:")
    print("Temperature range:", temperature_range)
    print("Cooling rate range:", cooling_rate_range)
    print("K range:", k_range)

    for temperature, cooling_rate, k in itertools.product(temperature_range, cooling_rate_range, k_range):
        # Mettre à jour la partition initiale avec le nouveau k
        current_partition = {node: random.randint(0, k) for node in graph.nodes()}
        
        # Appliquer le Simulated Annealing
        partition, score, _ = simulated_annealing(graph, current_partition, temperature, cooling_rate, iterations)
        
        # Enregistrer les meilleurs paramètres
        if score > best_score:
            best_score = score
            best_params = {
                'temperature': temperature,
                'cooling_rate': cooling_rate,
                'iterations': iterations,
                'k': k
            }
    
    return best_params, best_score


def plot_graph(G,partition):
    colors = [partition[node] for node in G.nodes()]
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=colors, cmap=plt.cm.Paired, with_labels=True)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Paired)
    sm.set_array([])
    cbar = plt.colorbar(sm, ticks=range(max(partition.values()) + 1))
    cbar.set_label('Clusters')
    plt.show()

def plot_adj(G,partition):
    ordered_nodes = sorted(partition.keys(), key=lambda x: partition[x])
    reordered_matrix = nx.to_numpy_array(G, nodelist=ordered_nodes)    
    plt.imshow(reordered_matrix, cmap='gray')
    plt.title("Reordered Adjacency Matrix")
    plt.show()