
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Function to read network from a file
def read_network_from_file(filename):
    # Creating an empty graph
    G = nx.Graph()
    # Opening the file
    with open(filename, 'r') as file:
        # Reading each line in the file
        for line in file:
            # Splitting the line into nodes and weight
            nodes = line.strip().split()
            # Checking if the line has all necessary components
            if len(nodes) == 3:
                node1, node2, weight = nodes
                # Adding an edge to the graph with specified weight
                G.add_edge(node1, node2, weight=float(weight))
    return G

# Function to calculate node degrees and plot degree distribution
def calculate_node_degrees(G):
    # Getting node degrees
    node_degrees = dict(G.degree())
    # Printing node degrees
    print("Node Degrees:", node_degrees)
    # Plotting degree distribution
    plt.figure(figsize=(8, 6))
    plt.hist(list(node_degrees.values()), bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Node Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    plt.show()

# Function to calculate number of connected components and visualize network graph
def calculate_connectivity(G):
    # Calculating number of connected components
    connected_components = nx.number_connected_components(G)
    # Printing number of connected components
    print("Number of Connected Components:", connected_components)
    # Visualizing network graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=700)
    plt.title("Network Graph")
    plt.show()

# Function to calculate shortest paths between all pairs of nodes and visualize them
def calculate_shortest_paths(G):
    # Finding shortest paths
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
    # Printing shortest paths
    print("Shortest Path Lengths:")
    for source in shortest_paths:
        for target in shortest_paths[source]:
            print(f"Shortest path between {source} and {target}: {shortest_paths[source][target]}")
    # Visualizing shortest paths
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color='skyblue', node_size=700, with_labels=True)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title("Shortest Paths Visualization")
    plt.show()

# Function to perform dynamic network analysis and plot results
def dynamic_network_analysis(data):
    # Plotting dynamic network analysis
    plt.figure(figsize=(8, 6))
    plt.plot(data['time'], data['num_edges'], marker='o')
    plt.xlabel('Time')
    plt.ylabel('Number of Edges')
    plt.title('Dynamic Network Analysis')
    plt.show()

# Function to perform community detection using machine learning algorithms and visualize communities
def machine_learning_analysis(G):
    # Detecting communities
    communities = nx.algorithms.community.greedy_modularity_communities(G)
    # Printing detected communities
    print("Communities detected:", communities)
    # Visualizing communities
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    node_colors = [0] * len(G)  
    for i, community in enumerate(communities):
        for node in community:
            node_index = node_mapping[node]
            node_colors[node_index] = i
    nx.draw(G, pos, node_color=node_colors, node_size=700, cmap=plt.cm.jet, with_labels=True)
    plt.title("Community Detection")
    plt.show()

# Function to visualize the network graph
def visualize_network(G):
    # Visualizing network graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=700)
    plt.title("Network Graph")
    plt.show()

# Function to calculate various centrality measures and visualize them
def calculate_centrality(G):
    # Calculating centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    # Printing centrality measures
    print("Degree Centrality:", degree_centrality)
    print("Betweenness Centrality:", betweenness_centrality)
    print("Closeness Centrality:", closeness_centrality)
    # Visualizing centrality measures
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=700, pos=nx.spring_layout(G))
    plt.title("Network Graph")
    plt.subplot(2, 2, 2)
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=[v * 3000 for v in degree_centrality.values()], pos=nx.spring_layout(G))
    plt.title("Degree Centrality")
    plt.subplot(2, 2, 3)
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=[v * 3000 for v in betweenness_centrality.values()], pos=nx.spring_layout(G))
    plt.title("Betweenness Centrality")
    plt.subplot(2, 2, 4)
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=[v * 3000 for v in closeness_centrality.values()], pos=nx.spring_layout(G))
    plt.title("Closeness Centrality")
    plt.tight_layout()
    plt.show()

# Function to detect topology of the network
def detect_topology(G):
    # Getting network parameters
    num_nodes = len(G.nodes())
    num_edges = len(G.edges())
    avg_degree = sum(dict(G.degree()).values()) / num_nodes
    # Determining topology based on parameters
    if num_edges == (num_nodes * (num_nodes - 1)) / 2:
        return "Fully Connected Topology"
    elif num_edges == num_nodes and avg_degree == 2:
        return "Ring Topology"
    elif num_edges == num_nodes - 1 and max(dict(G.degree()).values()) == (num_nodes - 1):
        return "Star Topology"
    elif num_edges == num_nodes - 1 and nx.is_tree(G):
        return "Tree Topology"
    elif avg_degree > (num_nodes - 1) / 2:
        return "Mesh Topology"
    else:
        return "Unknown Topology"

# Function to calculate network robustness under node and edge removal
def calculate_network_robustness(G):
    # Initializing results dictionary
    robustness_results = {'Remove Nodes': [], 'Remove Edges': []}
    num_nodes = len(G.nodes())
    num_edges = len(G.edges())
    G = G.to_undirected()
    # Getting size of largest connected component in original network
    largest_cc_size_original = len(max(nx.connected_components(G), key=len))
    robustness_results['Original'] = largest_cc_size_original / num_nodes
    # Calculating robustness under node removal
    for i in range(1, num_nodes):
        G_copy_nodes = G.copy()
        G_copy_edges = G.copy()
        # Removing nodes and edges
        for node in list(G.nodes())[:i]:
            G_copy_nodes.remove_node(node)
        largest_cc_size_nodes = len(max(nx.connected_components(G_copy_nodes), key=len))
        robustness_results['Remove Nodes'].append(largest_cc_size_nodes / num_nodes)
        for edge in list(G.edges())[:i]:
            G_copy_edges.remove_edge(*edge)
        largest_cc_size_edges = len(max(nx.connected_components(G_copy_edges), key=len))
        robustness_results['Remove Edges'].append(largest_cc_size_edges / num_nodes)
    # Plotting network robustness
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_nodes), robustness_results['Remove Nodes'], label='Remove Nodes', marker='o')
    plt.plot(range(1, num_nodes), robustness_results['Remove Edges'], label='Remove Edges', marker='o')
    plt.xlabel('Number of Nodes/Edges Removed')
    plt.ylabel('Fraction of Nodes in Giant Component')
    plt.title('Network Robustness Analysis')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main function to run the program
def main():
    # Main loop for file selection and analysis
    while True:
        root = tk.Tk()
        root.withdraw()
        # Asking user to select a file
        filename = filedialog.askopenfilename(title="Select File", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        # Exiting if no file selected
        if not filename:
            print("No file selected. Exiting.")
            return
        # Reading network from file
        G = read_network_from_file(filename)
        # Asking user if analysis should be performed
        analysis_choice = messagebox.askquestion("Analysis Choice", "Do you want to perform analysis on the network?")
        if analysis_choice == 'yes':
            # Loop for selecting analysis type
            while True:
                analysis_type = simpledialog.askstring("Analysis Type", "Enter analysis type (network visualization/degree/connectivity/shortest_paths/dynamic/ml/centrality/robustness/topology detector/select another file)")
                # Exiting if cancel is clicked
                if analysis_type is None:
                    return
                # Exiting loop if user chooses to select another file
                if analysis_type.lower() == 'select another file':
                    break 
                # Performing selected analysis
                if analysis_type == 'degree':
                    calculate_node_degrees(G)
                if analysis_type == 'network visualization':
                    visualize_network(G)
                elif analysis_type == 'connectivity':
                    calculate_connectivity(G)
                elif analysis_type == 'shortest_paths':
                    calculate_shortest_paths(G)
                elif analysis_type == 'dynamic':
                    dynamic_network_data = {'time': [1, 2, 3], 'num_edges': [10, 15, 20]}
                    dynamic_network_analysis(dynamic_network_data)
                elif analysis_type == 'centrality':
                    calculate_centrality(G)
                elif analysis_type == 'topology detector':
                    topology = detect_topology(G)
                    print("Detected Topology:", topology)
                elif analysis_type == 'ml':
                    machine_learning_analysis(G)
                elif analysis_type == 'robustness':
                    calculate_network_robustness(G)
                else:
                    print("Invalid analysis type entered.")
        else:
            print("No analysis performed.")
        # Informing user that analysis is complete
        messagebox.showinfo("Analysis Complete", "Analysis results have been printed to the console.")

# Running the main function if the script is executed directly
if __name__ == "__main__":
    main()
