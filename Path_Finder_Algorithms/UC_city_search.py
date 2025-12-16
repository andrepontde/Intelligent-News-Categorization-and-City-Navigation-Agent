import heapq

"""
Andre Pont - 23164034

UNIFORM COST SEARCH (UCS) ALGORITHM FOR CITY NAVIGATION

This algorithm was understood and adapted from the USC.ipynb lab code from week 3
the only difference is the dataset, since the same code will work for the graph that 
was given as a CA resource: CityMaps.ipynb (which I copied into a .py file to be able to use it as an import).

======================================================
Uniform Cost Search finds the optimal (lowest cost) path
between two nodes by exploring paths in order of increasing cost.
Similar to Dijkstra's algorithm but stops when goal is found but 
Does NOT use heuristics unlike A* 

Parameters:
- graph: Dictionary of {city: {neighbor: distance}} representing city connections
- start: Starting city name
- goal: Destination city name

Returns:
- cost: Total distance of the optimal route
- path: List of cities in optimal route from start to goal
"""
def uniform_cost_search(graph, start, goal):
    pq = [(0, start)]
    costs = {start: 0}
    parent = {start: None}
    
    # MAIN SEARCH LOOP
    while pq:
        current_cost, current_node = heapq.heappop(pq)
        if current_node == goal:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = parent[current_node]
            # Return optimal cost and path in correct order (start → goal)
            path.reverse()
            return path, current_cost
        # NEIGHBOR EXPLORATION
        for neighbor, edge_cost in graph[current_node].items():
            new_cost = current_cost + edge_cost
            if neighbor not in costs or new_cost < costs[neighbor]:
                costs[neighbor] = new_cost  
                parent[neighbor] = current_node
                heapq.heappush(pq, (new_cost, neighbor))
    # NO PATH FOUND: Return failure indicators
    return float('inf'), []  