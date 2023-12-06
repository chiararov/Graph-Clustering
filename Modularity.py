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

path='datasets/CA-HepTh.txt'
G= nx.read_edgelist(path, comments='#',delimiter='\t')

#print(spectral_clustering(G, k=50))
def modularity(G, clustering):
    m=G.number_of_edges()
    nc=len(set(clustering.values())) #number of clusters
    lc = {i: 0 for i in range(nc)} #number of edges in c
    dc = {i: 0 for i in range(nc)} #sum of degrees in c
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
# print(modularity(G,spectral_clustering(G,50)))

def compare_modularity_with_random(G,k):
    result1 = modularity(G,spectral_clustering(G,k))
    
    clustering={}

    for i,node in enumerate(G.nodes()):
        clustering[node]=randint(0,k-1)
    
    result2=modularity(G,clustering)

    return result1, result2

res1,res2=compare_modularity_with_random(G,50)
print('The modularity of the Spectral Clustering is:',res1,'The modularity of the Random Clustering is:',res2)


G = nx.Graph()

vertices_G = ['v1', 'v2', 'v3','v4','v5','v6','v7','v8']

G.add_nodes_from(vertices_G)

edges_G = [('v1', 'v2'),('v1', 'v3'),('v1', 'v4'),('v2', 'v3'),('v2', 'v4'),('v3', 'v4')
            ,('v3', 'v5'),('v4', 'v6'),('v5', 'v6'),('v5', 'v7'),('v5', 'v8'),('v6', 'v7')
            ,('v6', 'v8'),('v7', 'v8')]

G.add_edges_from(edges_G)

clustering1={}
clustering1['v1']=0
clustering1['v2']=0
clustering1['v3']=0
clustering1['v4']=0
clustering1['v5']=1
clustering1['v6']=1
clustering1['v7']=1
clustering1['v8']=1

clustering2={}
clustering2['v1']=0
clustering2['v2']=0
clustering2['v3']=1
clustering2['v4']=0
clustering2['v5']=1
clustering2['v6']=0
clustering2['v7']=1
clustering2['v8']=0   

result1=modularity(G,clustering1)
result2=modularity(G,clustering2)

print('The modularity of the first clustering is:',result1)
print('The modularity of the second clustering is:',result2)