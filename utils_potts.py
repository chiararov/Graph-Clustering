#### Plot graph with spin labels 
def plot_graph(G):
  plt.figure(figsize=(10,6))
  pos = nx.spring_layout(G)
  node_states = nx.get_node_attributes(G, 'spin')
  state_pos = {n: (x, y) for n, (x,y) in pos.items()}
  nx.draw(G,pos)
  nx.draw_networkx_labels(G, state_pos, labels=node_states, font_color='red')
  plt.show()


#### Very simple benchmark 

def generate_known_cluster(N_nodes, N_clusters, k_in, k_out):

  #Inputs: Desired number of nodes N_nodes, desired number of clusters N_clusters, Probability of intra-cluster connectivity k_in 
  # Probability of intra-cluster connectivity k_out
    
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

#### Sensitivity 
def sensitivity(G_ref, G_clustered):
    sensitivity = 0
    TP = 0
    FN = 0 
    tot = 0
    for i in range(len(G_ref.nodes())):
        for j in range(len(G_ref.nodes())):
            if i != j:
                #wrote the following like this to make it compatible with football graph
                n_i = list(G_ref.nodes())[i]
                n_j = list(G_ref.nodes())[j]
                if G_ref.nodes[n_i]["spin"] == G_ref.nodes[n_j]["spin"] and G_clustered.nodes[n_i]["spin"] == G_clustered.nodes[n_j]["spin"]:
                    TP+=1
                if G_ref.nodes[n_i]["spin"] == G_ref.nodes[n_j]["spin"] and G_clustered.nodes[n_i]["spin"] != G_clustered.nodes[n_j]["spin"]:
                    FN+=1
                    
    return TP/(TP+FN)

#### Specificity 

def specifity(G_ref, G_clustered):
    specificity = 0
    TN = 0
    FP = 0 
    tot = 0
    for i in range(len(G_ref.nodes())):
        for j in range(len(G_ref.nodes())):
            if i != j:
                 #wrote the following like this to make it compatible with football graph
                n_i = list(G_ref.nodes())[i]
                n_j = list(G_ref.nodes())[j]
                if (G_ref.nodes[n_i]["spin"] != G_ref.nodes[n_j]["spin"]) and G_clustered.nodes[n_i]["spin"] == G_clustered.nodes[n_j]["spin"]:
                    FP+=1
                if G_ref.nodes[n_i]["spin"] != G_ref.nodes[n_j]["spin"] and G_clustered.nodes[n_i]["spin"] != G_clustered.nodes[n_j]["spin"]:
                    TN+=1

    return TN/(TN+FP)
    

#### Thermal equilibrium step with Pott specific energy change

def metropolis(graph,J,gamma,beta,q):
    G = graph.copy()
    N_nodes = len(init_G.nodes)
    #Perform N_nodes**2 flips
    for node in range(N_nodes**2):
        
    
        #random node, wrote it like this to make it adaptable to the football graph 
        random_node = list(graph.nodes())[np.random.randint(N_nodes)]

        #choose a random spin within the q categories
        spin = np.random.randint(q)
        
        #Compute energy change due to interaction term
        d_interaction = 0
        for neigh in G.neighbors(random_node):
            d_interaction = d_interaction + J*(delta(G.nodes[random_node]["spin"],G.nodes[neigh]["spin"]) - delta(spin,G.nodes[neigh]["spin"]))
        
        #Compute energy change due to the other term 
        n_k = number_spin(G,G.nodes[random_node]["spin"])
        n_spin = number_spin(G,spin)
        dE = d_interaction + gamma*(n_spin - n_k)
    
        
        if dE < 0:
            G.nodes[random_node]["spin"] = spin
           
            
        elif np.random.rand() < np.exp(-beta*dE):
            G.nodes[random_node]["spin"] = spin
            
        
    return(G)

#### Simulated annealing 

def monte_carlo_pott(graph, J,gamma, beta, q, eq_steps, mc_steps, alpha):
    
    graph = random_initialization(graph,q)
    N_nodes = len(graph.nodes)
    co_matrix = np.zeros((N_nodes, N_nodes))
    E = [] #List of energies
    B = [] #List of inverse temperatures
    #equilibrium steps 
    
        
    for i in range(mc_steps):
        print("Annealing step: " + str(i))
        co_matrix = coappearance(graph,co_matrix)
        for _ in range(eq_steps):
            
            G = metropolis(graph, J, gamma, beta, q)
            graph = G.copy()
          
        E.append(calc_energy(graph, J, gamma, q))
        beta = beta/alpha
        B.append(beta)

            
    
    return(G,B,E, co_matrix)
