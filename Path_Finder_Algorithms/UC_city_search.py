import heapq

def uniform_cost_search(graph, start, goal):
    # Priority queue to store (cost, node) and initialize with the start node
    pq = [(0, start)]
    
    # Dictionary to store the cost to reach each node, initialize start node cost as 0
    costs = {start: 0}
    
    # Dictionary to store the parent of each node for path reconstruction
    parent = {start: None}
    
    while pq:
        current_cost, current_node = heapq.heappop(pq)
        
        # If we reach the goal, return the cost and the path
        if current_node == goal:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = parent[current_node]
            path.reverse()  # Reverse the path to get the correct order
            return current_cost, path
        
        # Explore neighbors of the current node
        for neighbor, edge_cost in graph[current_node].items():
            new_cost = current_cost + edge_cost
            
            # If the new path to the neighbor is cheaper, update and push to queue
            if neighbor not in costs or new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                parent[neighbor] = current_node
                heapq.heappush(pq, (new_cost, neighbor))
    
    return float('inf'), []  # Return infinite cost and empty path if goal is not reachable