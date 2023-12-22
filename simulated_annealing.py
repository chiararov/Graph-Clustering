import itertools
import matplotlib.pyplot as plt
import networkx as nx
import random
import math
from Utils import *
from tqdm import tqdm

def modularity(G, clustering):
    '''
    Calculates the modularity of a graph G based on the clustering, handled as a dictionary.
    Returns the modularity value
    '''
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


def simulated_annealing(graph, k, temperature, cooling_rate, iterations):
    '''
    Algorithm that maximimizes the modularity. k is the number of clusters.
    It returns:
    - The final partition
    - The final modularity score
    - The evolution of the modularity along the iterations
    '''
    current_partition = {node: random.randint(1,k) for node in graph.nodes()}
    current_score = modularity(graph, current_partition)

    scores=[]
    n=graph.number_of_nodes()
    for _ in range(n*iterations):
        for _ in range(n):
            new_partition = perturb_partition(current_partition)
            new_score = modularity(graph, new_partition)

            if new_score > current_score:
                current_partition = new_partition
                current_score = new_score

            proba=math.log(random.random() + 1e-10 )
            comparison=(new_score - current_score) / temperature
            
            if proba < comparison:
                current_partition = new_partition
                current_score = new_score
            
            scores.append(current_score)
        temperature *= 1 - cooling_rate

    return current_partition, current_score,scores

def perturb_partition(partition):
    '''Changes the cluster of a node'''
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
    '''
    Realises finds the optimised set of parameters that maximises the modularity.
    Returns a set of parameters
    '''
    best_score = -float('inf')
    best_params = {}

    for temperature in temperature_range:
        for cooling_rate in cooling_rate_range:
            partition, score, _ = simulated_annealing(graph, k, temperature, cooling_rate, iterations)
            if score > best_score:
                best_score = score
                best_params = {
                    'temperature': temperature,
                    'cooling_rate': cooling_rate,
                }
    
    return best_params, best_score

def find_para(G,k,temperature_range,cooling_rate_range,iterations,threshold):
    '''
    Because of the modularity's behaviour, we introduce an other type of grid search:
    We calculate the optimised set of parameters and return the set, the clustering, the final modularity score and a track of the modularity values during the iterations.
    '''
    Good=[]
    for temperature in temperature_range:
        for cooling_rate in cooling_rate_range:
            final_partition, final_score, scores = simulated_annealing(G, k, temperature, cooling_rate, iterations)
            S=set(final_partition.values())
            if final_score>threshold:
                Good.append((temperature,cooling_rate,final_partition, final_score, scores))
    return Good
