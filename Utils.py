"""
Graph Clustering - MVA - Dec 2023
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans

# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    ##################
    A= nx.adjacency_matrix(G)
    
    D_inv=diags([1/G.degree(node) for node in G.nodes()])
    
    calcul=np.dot(D_inv,A)
    L= eye(calcul.shape[0])- calcul

    evals, evecs= eigs(L, k=k, which = 'SR')
    evecs=np.real(evecs)
    kmeans=KMeans(n_clusters=k,n_init=10).fit(evecs)

    clustering={}
    for i,node in enumerate(G.nodes()):
        clustering[node]=kmeans.labels_[i]
    return clustering

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
    for i in range(nc):
        Q+=lc[i]/m-(dc[i]/(2*m))**2

    return Q

def compare_modularity_with_random(G,clustering):
    result1 = modularity(G,clustering)

    k=len(set(clustering.values()))
    clustering_random={}
    for i,node in enumerate(G.nodes()):
        clustering_random[node]=randint(0,k-1)
    result2=modularity(G,clustering_random)

    return result1, result2

