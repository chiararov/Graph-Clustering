import matplotlib.pyplot as plt
import networkx as nx
import random
import math
from Utils import modularity

def simulated_annealing(graph, initial_partition, temperature, cooling_rate, iterations):
    current_partition = initial_partition.copy()
    current_score = modularity(graph, current_partition)

    best_partition = current_partition.copy()
    best_score = current_score

    for _ in range(iterations):
        new_partition = perturb_partition(current_partition)
        new_score = modularity(graph, new_partition)

        if new_score > current_score or random.random() < math.exp((new_score - current_score) / temperature):
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


# Exemple d'utilisation
if __name__ == "__main__":
    path="Datasets/CA-HepTh.txt"
    G= nx.read_edgelist(path, comments='#',delimiter='\t') 
    # Initialiser une partition initiale
    initial_partition = {node: random.randint(0,11) for node in G.nodes()}

    # Paramètres du Simulated Annealing
    initial_temperature = 1.0
    cooling_rate = 0.01
    iterations = 100

    # Appliquer le Simulated Annealing
    final_partition, final_score = simulated_annealing(G, initial_partition, initial_temperature, cooling_rate, iterations)

    # Afficher les résultats
    print("Final Partition:", final_partition)
    print("Final Modularity Score:", final_score)
