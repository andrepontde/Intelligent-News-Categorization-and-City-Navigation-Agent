
"""
Andre Pont - 23164034

A* SEARCH ALGORITHM FOR CITY NAVIGATION

This algorithm was understood and adapted from the A_star_Search lab code from week 3
the only difference is the dataset, since the same code will work for the graph that 
was given as a CA resource: CityMaps.ipynb (which I copied into a .py file to be able to use it as an import).

=====================================
A* finds the optimal path between two nodes by combining Dijkstra's Algorithm and Greedy Best-First Search

Parameters:
- graph: Dictionary of {city: {neighbor: distance}} representing city connections
- start: Starting city name
- goal: Destination city name  
- heuristics: Dictionary of {city: estimated_distance_to_goal}

Returns:
- path: List of cities in optimal route from start to goal
- path_cost: Total distance of the optimal route
"""
import heapq

def a_star_search(graph, start, goal, heuristics): 
    open_set = [(heuristics[start], start)]
    g_score = {start: 0}
    f_score = {start: heuristics[start]}
    came_from = {start: None}

    # MAIN SEARCH LOOP
    while open_set:
        nan, current = heapq.heappop(open_set)

        # GOAL CHECK - Early termination when destination is reached
        if current == goal:
            path = []           # Will store the sequence of cities in the optimal route
            path_cost = 0       # Will accumulate the total distance of the route
            while current:
                path.append(current)
                # Add the cost of the edge we just traced back
                # Skip this for the start node since which has no parent
                if came_from[current] is not None:
                    # Add distance from parent city to current city
                    path_cost += graph[came_from[current]][current]
                
                # Move to the parent city
                current = came_from[current]
            
            # Return path in correct order (start → goal) and total cost
            return path[::-1], path_cost

        # NEIGHBOR EXPLORATION - Examine all cities directly connected to current city
        for neighbor, cost in graph[current].items():
            # Calculate potential new cost to reach this neighbor through current city
            tentative_g_score = g_score[current] + cost
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                # UPDATE PATH INFORMATION - Record the better route

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristics[neighbor]
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    # NO PATH FOUND - Return failure indicators
    return None, float('inf')