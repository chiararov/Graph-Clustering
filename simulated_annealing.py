import itertools
import matplotlib.pyplot as plt
import networkx as nx
import random
import math
from Utils import *
from tqdm import tqdm

def modularity(G, clustering):
    m=G.number_of_edges()
    unique_clusters=set(clustering.values())
    nc=len(unique_clusters) #number of clusters
    lc = {cluster: 0 for cluster in unique_clusters} #number of edges in c
    dc = {cluster: 0 for cluster in unique_clusters} #sum of degrees in c
    already=set() #list of the nodes already treated
    for edge in G.edges():
        node1,node2=edge
        cluster1=clustering.get(node1)
        cluster2=clustering.get(node2)

        if cluster1==cluster2:
            lc[cluster1]+=1
            if not node1 in already:
                dc[cluster1]+= G.degree(node1)
                already.add(node1)
            if not node2 in already:
                dc[cluster1]+= G.degree(node2)
                already.add(node2)
    Q=0
    for i in unique_clusters:
        Q+=lc[i]/m-(dc[i]/(2*m))**2

    return Q


def simulated_annealing(graph, initial_partition, temperature, cooling_rate, iterations):
    current_partition = initial_partition.copy()
    current_score = modularity(graph, current_partition)

    best_partition = current_partition.copy()
    best_score = current_score
    scores=[]
    n=graph.number_of_nodes()
    for _ in range(n*iterations):
        for _ in range(n//5):
            new_partition = perturb_partition(current_partition)
            new_score = modularity(graph, new_partition)

            if new_score > current_score:
                current_partition = new_partition
                current_score = new_score

            #pour eviter le depassement numerique, on change la condition classique
            proba=math.log(random.random() + 1e-10 )
            comparison=(new_score - current_score) / temperature
            
            if proba < comparison:
                # best_partition = new_partition
                # best_score = new_score
                current_partition = new_partition
                current_score = new_score
            
            # scores.append(best_score)
            scores.append(current_score)
        temperature *= 1 - cooling_rate

    # return best_partition, best_score,scores
    return current_partition, current_score,scores

def perturb_partition(partition):
    perturbed_partition = partition.copy()
    node = random.choice(list(partition.keys()))
    available_communities = set(partition.values()) - {partition[node]}
    if not available_communities:
        new_community = random.choice(list(set(partition.values())))
    else:
        new_community = random.choice(list(available_communities))
    
    perturbed_partition[node] = new_community
    return perturbed_partition


def grid_search(graph, temperature_range, cooling_rate_range, iterations, k):
    best_score = -float('inf')
    best_params = {}

    # for temperature, cooling_rate in itertools.product(temperature_range, cooling_rate_range):
    for temperature in temperature_range:
        for cooling_rate in cooling_rate_range:
            current_partition = {node: random.randint(1, k) for node in graph.nodes()}
            partition, score, _ = simulated_annealing(graph, current_partition, temperature, cooling_rate, iterations)
            if score > best_score:
                best_score = score
                best_params = {
                    'temperature': temperature,
                    'cooling_rate': cooling_rate,
                }
    
    return best_params, best_score

def plot_graph(G,partition):
    colors = [partition[node] for node in G.nodes()]
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=colors, cmap=plt.cm.Paired, with_labels=False)
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

# G = generate_known_cluster(100,5,0.7,0.1)
# k=6
# temperature=5
# cooling_rate=0.07
# iterations=5
# initial_partition = {node: random.randint(0,k) for node in G.nodes()}

# final_partition, final_score, scores = simulated_annealing(G, initial_partition, temperature, cooling_rate, iterations)

# # print("Final Partition:", final_partition)
# print("Final Modularity Score:", final_score)

# iterations = 5
# temperature_range = np.arange(3, 7, 1)
# cooling_rate_range = np.arange(0.01,0.2,0.02)
# k = 5
# best_params, best_score = grid_search(G, temperature_range, cooling_rate_range, iterations, k)