##### Compute delta function 
def delta(i,j):
  if i==j: 
      return 1
  else: 
      return 0 


##### Compute the number of nodes having spin q 
def number_spin(G,q):
    selected_nodes = [n for n,value in G.nodes(data=True) if value['spin'] == q ]  
    return len(selected_nodes)

##### Calculate the energy of a graph given a configuration 

def calc_energy(graph, J,gamma,q):
    interaction_term = 0
    for edge in graph.edges:
        node_i = graph.nodes[edge[0]]["spin"] 
        node_j = graph.nodes[edge[1]]["spin"] 
        interaction_term += -J*delta(node_i, node_j)
    other_term = 0
    
    for i in range(q):
        n_s = number_spin(graph,i)
        other_term += gamma*(n_s-1)*n_s*0.5
        
    return interaction_term + other_term

#### Coappearance matrix 

def coappearance(G,co_matrix):
    N_nodes = len(init_G.nodes)
    for i in range(N_nodes):
        for j in range(N_nodes):
            if G.nodes[i]["spin"] == G.nodes[j]["spin"]:
                co_matrix[i,j]+=1
    return(co_matrix)
       
