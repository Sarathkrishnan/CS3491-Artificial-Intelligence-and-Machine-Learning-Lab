import heapq


def aStarAlgo(start_node, stop_node):
    # Priority queue for the open set
    open_set = []
    heapq.heappush(open_set, (0, start_node))
    closed_set = set()

    # g represents the cost to reach each node
    g = {start_node: 0}

    # Parents dictionary to reconstruct the path
    parents = {start_node: None}

    while open_set:
        # Get the node with the smallest f(n) = g(n) + h(n)
        _, current_node = heapq.heappop(open_set)

        # If the destination node is reached
        if current_node == stop_node:
            path = []
            while current_node:
                path.append(current_node)
                current_node = parents[current_node]
            path.reverse()
            return path

        # Add the current node to the closed set
        closed_set.add(current_node)

        # Process neighbors
        for neighbor, weight in get_neighbors(current_node):
            if neighbor in closed_set:
                continue

            tentative_g = g[current_node] + weight
            if neighbor not in g or tentative_g < g[neighbor]:
                g[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor)
                heapq.heappush(open_set, (f, neighbor))
                parents[neighbor] = current_node

    return None  # Path does not exist


# Function to get the neighbors of a node
def get_neighbors(v):
    return Graph_nodes.get(v, [])


# Heuristic function (h(n)): Estimated cost to reach the goal
def heuristic(n):
    H_dist = {
        "A": 11,
        "B": 6,
        "C": 5,
        "D": 7,
        "E": 3,
        "F": 6,
        "G": 5,
        "H": 3,
        "I": 1,
        "J": 0,
    }
    return H_dist.get(n, float("inf"))


# Graph definition
Graph_nodes = {
    "A": [("B", 6), ("F", 3)],
    "B": [("A", 6), ("C", 3), ("D", 2)],
    "C": [("B", 3), ("D", 1), ("E", 5)],
    "D": [("B", 2), ("C", 1), ("E", 8)],
    "E": [("C", 5), ("D", 8), ("I", 5), ("J", 5)],
    "F": [("A", 3), ("G", 1), ("H", 7)],
    "G": [("F", 1), ("I", 3)],
    "H": [("F", 7), ("I", 2)],
    "I": [("E", 5), ("G", 3), ("H", 2), ("J", 3)],
}

# Example usage
print("Output (A*):")
result = aStarAlgo("A", "J")
if result:
    print("Path found:", result)
else:
    print("Path does not exist!")
