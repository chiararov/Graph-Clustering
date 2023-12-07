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

    for _ in range(iterations):
        new_partition = perturb_partition(current_partition)
        new_score = modularity(graph, new_partition)

        #pour eviter le depassement numerique, on change la condition classique
        if new_score > current_score or math.log(random.random() + 1e-10 ) < (new_score - current_score) / temperature:
            current_partition = new_partition
            current_score = new_score

        if new_score > best_score:
            best_partition = new_partition
            best_score = new_score

        temperature *= 1 - cooling_rate

    return best_partition, best_score

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

def plot(G,partition):

    ordered_nodes = sorted(partition.keys(), key=lambda x: partition[x])

    reordered_matrix = nx.to_numpy_array(G, nodelist=ordered_nodes)
    
    # Plot adjacency matrix
    plt.imshow(reordered_matrix, cmap='gray')
    plt.title("Reordered Adjacency Matrix")
    plt.show()

    # Draw the graph
    plt.title('Generated graph')
    nx.draw(G)
    plt.show()


path="Datasets/CA-HepTh.txt"
G= nx.read_edgelist(path, comments='#',delimiter='\t')

# Paramètres du Simulated Annealing
initial_temperature = 2.0
cooling_rate = 0.01
iterations = 1000
k=10

# Initialiser une partition initiale
initial_partition = {node: random.randint(0,k) for node in G.nodes()}

# # Appliquer le Simulated Annealing
# final_partition, final_score = simulated_annealing(G, initial_partition, initial_temperature, cooling_rate, iterations)

# # Afficher les résultats
# print("Final Partition:", final_partition)
# print("Final Modularity Score:", final_score)
# print(plot(G,final_partition))


def grid_search(graph, initial_partition, temperature_range, cooling_rate_range, iterations, k_range):
    best_score = -float('inf')
    best_params = {}
    print("Debug in grid_search:")
    print("Temperature range:", temperature_range)
    print("Cooling rate range:", cooling_rate_range)
    print("Iteration range:", iteration_range)
    print("K range:", k_range)
    for temperature, cooling_rate, k in itertools.product(temperature_range, cooling_rate_range, k_range):
        # Mettre à jour la partition initiale avec le nouveau k
        current_partition = {node: random.randint(0, k) for node in graph.nodes()}
        
        # Appliquer le Simulated Annealing
        partition, score = simulated_annealing(graph, current_partition, temperature, cooling_rate, iterations)
        
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

# Définition des gammes de paramètres
# temperature_range = [0.5, 1.0, 2.0]
# cooling_rate_range = [0.01, 0.05, 0.1]
# iteration_range = [500, 1000, 1500]
# k_range = [3, 5, 7]

temperature_range = np.arange(1.5, 3.5, 0.5)
cooling_rate_range = np.arange(0.05, 0.5, 0.05)
iteration_range = 10000
k_range = np.arange(3, 8, 1) 

# Appliquer la recherche en grille
best_params, best_score = grid_search(G, initial_partition, temperature_range, cooling_rate_range, iteration_range, k_range)

print("Meilleurs paramètres:", best_params)
print("Meilleur score de modularité:", best_score)
