# Using NetworkX to work with GraphML files
import networkx as nx
import matplotlib.pyplot as plt

# Load the GraphML file
G = nx.read_graphml("your_file.graphml")

# Visualize the graph
plt.figure(figsize=(12, 8))
nx.draw(G, with_labels=True, node_color="skyblue", node_size=1500, edge_color="gray")
plt.show()
