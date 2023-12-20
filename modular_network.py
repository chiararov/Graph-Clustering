def modular_network(n_cliques,nodes_per_click):
  #Inputs: nombres de cliques et nombres de nodes par cliques, pour avoir le phénomène de résolution limite il faut choisir n_cliques et nodes_per_click 
  # tel que , n_cliques > nodes_per_clique*(nodes_per_clique-1)+2 
  # nodes_per_clique = 5 et n_cliques = 30 par exemple 
    net = nx.Graph()
    #Create a graph of n_cliques modules with equal number of nodes in each
    for node in range(n_cliques*nodes_per_click):
        net.add_node(node)
        cluster = node//nodes_per_click
        net.nodes[node]["spin"] = cluster

    #Convert these modules into cliques
    for k in range(n_cliques):
        selected_nodes = [n for n,v in net.nodes(data=True) if v['spin'] == k]
        for i in range(nodes_per_click):
            for j in range(i+1,nodes_per_click):
                net.add_edge(selected_nodes[i],selected_nodes[j])
      #Connect each clique with one random link
    for k in range(n_cliques-1):
        selected_i =  [n for n,v in net.nodes(data=True) if v['spin'] == k]
        selected_i_1 = [n for n,v in net.nodes(data=True) if v['spin'] == k+1]
        nodes_i = selected_i[np.random.randint(nodes_per_click)]
        nodes_i_1 = selected_i_1[np.random.randint(nodes_per_click)]
        net.add_edge(nodes_i,nodes_i_1)

    #Connect the first and last clique
    selected_i =  [n for n,v in net.nodes(data=True) if v['spin'] == n_cliques-1]
    selected_i_1 = [n for n,v in net.nodes(data=True) if v['spin'] == 0]
    nodes_i = selected_i[np.random.randint(nodes_per_click)]
    nodes_i_1 = selected_i_1[np.random.randint(nodes_per_click)]
    net.add_edge(nodes_i,nodes_i_1)
    
    graph = net.copy()
    
    #Single clustering
    clusters_single = {}
    for i in range(n_cliques*nodes_per_click):
        clusters_single[i] = net.nodes[i]["spin"] 
    
    
    #Pair clustering 
    for node in range(n_cliques*nodes_per_click):
        net.add_node(node)
        cluster = node//(2*nodes_per_click)
        net.nodes[node]["spin"] = cluster
    
    clusters_pairs = {}
    for i in range(n_cliques*nodes_per_click):
        clusters_pairs[i] = net.nodes[i]["spin"] 
    return graph, clusters_single, clusters_pairs
