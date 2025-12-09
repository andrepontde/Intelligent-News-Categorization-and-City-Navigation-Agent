import heapq

def a_star_search(graph, start, goal, heuristics):
    open_set = [(heuristics[start], start)]
    g_score = {start: 0}
    f_score = {start: heuristics[start]}
    came_from = {start: None}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            path_cost = 0
            while current:
                path.append(current)
                if came_from[current] is not None:
                    path_cost += graph[came_from[current]][current]
                current = came_from[current]
            return path[::-1], path_cost

        for neighbor, cost in graph[current].items():
            tentative_g_score = g_score[current] + cost

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristics[neighbor]
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None, float('inf')