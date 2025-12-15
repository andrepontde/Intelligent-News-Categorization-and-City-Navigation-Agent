"""
Andre Pont - 23164034

TRAVELING SALESMAN PROBLEM (TSP) WITH GENETIC ALGORITHM

The genetic algorithm, retrieved from lab 4 Shortest_Path_Problem code, was as well as  most algorithms from this 
assessment, was understood and repoupoused to be able to work with a different dataset,
but also with the small change that the city phoenix has to always be the first and 
last destination, which did require the most changes out of all of the other lab codes.


======================================================

GENETIC ALGORITHM APPROACH:
KEY MODIFICATIONS FROM STANDARD TSP:
 - Fixed Start/End Point: Phoenix always begins and ends the route
 - Dynamic City Selection: Users choose subset of cities to visit
 - Filtered Distance Matrix: Only includes selected cities to optimize computation
 - Modified Fitness Function: Accounts for round trip starting/ending at Phoenix
 - City Selection Interface: Interactive menu for choosing offices to visit

RETURNS:
- best_path: Optimal order for visiting selected cities
- best_distance: Total distance of optimal route including return to Phoenix
"""

import random
import numpy as np
import matplotlib.pyplot as plt

distance_matrix = [
    [0, 370, 355, 300],    # Phoenix
    [370, 0, 120, 270],    # Los Angeles  
    [355, 120, 0, 330],    # San Diego
    [300, 270, 330, 0]     # Las Vegas
]

cities = ["Phoenix", "Los Angeles", "San Diego", "Las Vegas"]

# Function to prompt user to select cities to visit
# Interactive city selection to allow users to choose which offices to visit
# This replaces the previous hardcoded approach and makes the system flexible
def select_cities_to_visit():
    # Only Los Angeles, San Diego, and Las Vegas are available for selection
    # Phoenix is automatically included as the starting/ending point
    available_cities = ["Los Angeles", "San Diego", "Las Vegas"]
    print("Welcome to the TSP Route Optimizer!")
    print("Phoenix (Head Office) is your starting and ending point.")
    print("\nAvailable offices to visit:")
    
    # Display numbered menu for easy selection
    for i, city in enumerate(available_cities, 1):
        print(f"{i}. {city}")
    
    selected_cities = []
    while True:
        try:
            choice = input("\nEnter the number of a city to add to your route (or 'done' to finish): ").strip()
            if choice.lower() == 'done':
                # Ensure at least one city is selected (TSP requires visiting at least one location)
                if len(selected_cities) == 0:
                    print("Please select at least one city to visit.")
                    continue
                break
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_cities):
                city = available_cities[choice_num - 1]
                # Prevent duplicate city selections
                if city not in selected_cities:
                    selected_cities.append(city)
                    print(f"Added {city} to your route.")
                else:
                    print(f"{city} is already in your route.")
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number or 'done'.")
    
    return selected_cities

# Function to create filtered distance matrix and city indices
# Change from original code: Dynamic matrix filtering based on user selection
# This allows the GA to work with only the selected cities instead of all cities
def create_filtered_data(selected_city_names):
    # Always include Phoenix as index 0
    # This ensures Phoenix is always the reference point for the TSP
    filtered_cities = ["Phoenix"] + selected_city_names
    city_to_index = {city: cities.index(city) for city in filtered_cities}
    
    # Create filtered distance matrix containing only selected cities
    filtered_matrix = []
    for city1 in filtered_cities:
        row = []
        for city2 in filtered_cities:
            orig_idx1 = cities.index(city1)
            orig_idx2 = cities.index(city2)
            row.append(distance_matrix[orig_idx1][orig_idx2])
        filtered_matrix.append(row)
    
    return filtered_matrix, filtered_cities

def initialize_population(size, num_cities):
    population = []
    for _ in range(size):
        individual = random.sample(range(num_cities), num_cities)
        population.append(individual)
    return population

# Function to calculate the total distance of a path
# Change from original code: Simplified to work with Phoenix as fixed start/end point
# Original function required separate start_city and end_city parameters
def calculate_fitness(individual, distance_matrix):
    # Phoenix is always at index 0, path starts and ends there
    total_distance = 0    
    # Start from Phoenix to first city in path
    if len(individual) > 0:
        total_distance += distance_matrix[0][individual[0]]
        for i in range(len(individual) - 1):
            total_distance += distance_matrix[individual[i]][individual[i + 1]]
        
        total_distance += distance_matrix[individual[-1]][0]
    
    return total_distance

# Function to select parents based on their fitness
def select_parents(population, fitness_scores):
    fitness_total = sum(fitness_scores)
    probabilities = [1 - (fitness / fitness_total) for fitness in fitness_scores]
    probabilities = np.array(probabilities) / np.sum(probabilities)
    return random.choices(population, weights=probabilities, k=2)

# Function to perform crossover
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size

    child[start:end] = parent1[start:end]

    current_pos = 0
    for gene in parent2:
        if gene not in child:
            while child[current_pos] != -1:
                current_pos += 1
            child[current_pos] = gene
    return child

# Function to perform mutation by swapping two cities
def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]

# Main Genetic Algorithm to find the shortest path
# Original code change: Adapted to work with dynamic city selection and Phoenix constraints
# Original function required start_city and end_city parameters, now Phoenix is hardcoded as both
def genetic_algorithm(distance_matrix, population_size, generations, mutation_rate):
    num_cities_to_visit = len(distance_matrix) - 1
    
    if num_cities_to_visit == 0:
        return [], 0 
    population = []
    for x in range(population_size):
       
        individual = random.sample(range(1, len(distance_matrix)), num_cities_to_visit)
        population.append(individual)

    best_individual = None
    best_distance = float('inf')

    for generation in range(generations):
        fitness_scores = [calculate_fitness(individual, distance_matrix) for individual in population]
        next_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitness_scores)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            next_population.extend([child1, child2])

        population = sorted(next_population, key=lambda x: calculate_fitness(x, distance_matrix))[:population_size]

        # Track the best individual found so far
        current_best_individual = population[0]
        current_best_distance = calculate_fitness(current_best_individual, distance_matrix)
        if current_best_distance < best_distance:
            best_individual = current_best_individual
            best_distance = current_best_distance

    return best_individual, best_distance

