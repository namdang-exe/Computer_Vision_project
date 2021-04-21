import numpy as np
import random
import networkx as nx
from IPython.display import Image
import matplotlib.pyplot as plt

# Load the graph
G_karate = nx.karate_club_graph()
# Find key-values for the graph
pos = nx.spring_layout(G_karate)
# Plot the graph
nx.draw(G_karate, cmap = plt.get_cmap('rainbow'), with_labels=True, pos=pos)
plt.show()