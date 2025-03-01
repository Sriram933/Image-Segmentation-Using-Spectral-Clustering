# Image Segmentation using Graph Clustering

This project constructs a graph from an image, where each pixel is treated as a node, and edges are created based on pixel adjacency and intensity similarity. The graph is then used for spectral clustering to segment the image.

## Features
- Converts an image into a **graph representation**
- Constructs an **affinity matrix** based on pixel similarity
- Uses **eigenvalue decomposition** for spectral clustering
- Segments an image into distinct regions
- Visualizes the segmented output

## The explanation and code for the above features is as follows:

### 1) Graph Construction
- Nodes: Each pixel (x, y) in the image is treated as a node.
- Edges: Nodes (pixels) are connected based on intensity similarity and spatial distance.
- Weight Calculation: Uses a Gaussian function to measure similarity based on RGB values and inverse distance.


```python

def calculate_weight(value1, value2, pos1, pos2, sigma=20):
    """
    Calculate the weight between two pixels using intensity values and positions.
    
    Parameters:
    - value1, value2: RGB intensity values of the pixels.
    - pos1, pos2: (x, y) coordinates of the pixels.
    - sigma: Standard deviation for the Gaussian function.
    
    Returns:
    - weight: Computed weight based on intensity and spatial distance.
    """
    intensity_diff_squared = sum((a - b) ** 2 for a, b in zip(value1, value2))
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
    if distance == 0:
        distance = 1e-5
    weight = math.exp(-intensity_diff_squared / (2 * sigma ** 2)) * (1 / distance)
    return weight


G = nx.Graph()
img = np.random.randint(0, 256, (60, 60, 3))  
# Create nodes (each pixel as a node)
for x in range(60):
    for y in range(60):
        node_id = (x, y)
        values = tuple(img[x, y])
        G.add_node(node_id, values=values)

# Create edges based on intensity and spatial similarity
for x in range(60):
    for y in range(60):
        for px in range(x, 60):
            for py in range(60):
                if px == x and py <= y:
                    continue  # Avoid duplicate edges
                node_id = (x, y)
                other_node_id = (px, py)
                weight = calculate_weight(G.nodes[node_id]['values'], 
                                          G.nodes[other_node_id]['values'],
                                          node_id,
                                          other_node_id)
                G.add_edge(node_id, other_node_id, weight=weight)

```
### 2) Weight and Degree Matrices

- Weight Matrix (W): Stores computed weights between connected pixels.

- Degree Matrix (D): Stores the sum of all edge weights for each node.

```python
size = 60
W = np.zeros((size * size, size * size))
D = np.zeros((size * size, size * size))

for i, node_i in enumerate(G.nodes()):
    for j, node_j in enumerate(G.nodes()):
        if G.has_edge(node_i, node_j):
            W[i, j] = G[node_i][node_j]['weight']
            D[i, i] += G[node_i][node_j]['weight']

```

### 3) Spectral Clustering

- The adjacency matrix (W) is used to apply Spectral Clustering.
- The algorithm groups pixels into two clusters based on spectral properties.
- The clustering results are reshaped into an image-like structure.

```python
def cluster_graph(adjacency_matrix):
    """
    Performs spectral clustering on the given adjacency matrix.
    
    Parameters:
    - adjacency_matrix: Weighted adjacency matrix of the graph.
    
    Returns:
    - labels: Cluster labels assigned to each node.
    """
    adjacency_matrix = np.array(adjacency_matrix)
    spectral = SpectralClustering(n_clusters=2, affinity='precomputed', assign_labels='kmeans')
    labels = spectral.fit_predict(adjacency_matrix)
    return labels

clusters = cluster_graph(W)
reshaped_assignments = clusters.reshape((60, 60))
```

