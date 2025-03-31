import numpy as np

class Ant:
    def __init__(self, alpha, beta, gamma, num_cities):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.food = False
        self.path = []
        self.distance = 0
        self.visited = np.zeros(num_cities, dtype=bool)

    def choose_next_city(self, current_city, pheromones, distances):
        probabilities = []
        cities = []
        
        for city in range(len(pheromones)):
            if not self.visited[city] and distances[current_city][city] > 0:
                tau = pheromones[current_city][city] ** self.alpha
                eta = (1 / distances[current_city][city]) ** self.beta
                probabilities.append(tau * eta)
                cities.append(city)

        if probabilities:
            probabilities = np.array(probabilities) / sum(probabilities)  # Normalize probabilities
            return np.random.choice(cities, p=probabilities)

        return None  # If no valid city is left


    def travel(self, start_city, pheromones, distances):
        self.path = [start_city]
        self.visited[start_city] = True
        current_city = start_city

        while len(self.path) < len(pheromones):
            next_city = self.choose_next_city(current_city, pheromones, distances)
            if next_city is None:
                break
            self.path.append(next_city)
            self.visited[next_city] = True
            self.distance += distances[current_city][next_city]
            current_city = next_city

        self.path.append(start_city)
        self.distance += distances[current_city][start_city]

    def deposit_pheromones(self, pheromones):
        for i in range(len(self.path) - 1):
            pheromones[self.path[i]][self.path[i + 1]] += 1 / self.distance

class Environment:
    def __init__(self, num_cities, alpha, beta, gamma, num_ants, positions):
        self.num_cities = num_cities
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.pheromones = np.ones((num_cities, num_cities))
        self.positions = positions
        self.distances = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    self.distances[i][j] = np.linalg.norm(
                        np.array(self.positions[i]) - np.array(self.positions[j])
                    )
        np.fill_diagonal(self.distances, np.inf)
        self.ants = [Ant(alpha, beta, gamma, num_cities) for _ in range(num_ants)]

    def evaporate_pheromones(self):
        self.pheromones *= (1 - self.gamma)
        
        print(f"Pheromone matrix after evaporation:\n{self.pheromones}")
    
    def simulate(self, num_iterations, track_progress=False, patience = 20):
        best_distance = float('inf')
        best_path = None
        distance_progress = [] if track_progress else None
        
        no_improvement = 0
    
        for iteration in range(num_iterations):
            pheromone_delta = np.zeros_like(self.pheromones)  # Temporary matrix for Δτij
    
            for ant in self.ants:
                ant.visited[:] = False  # Reset visited status
                ant.path = []
                ant.distance = 0  # Reset distance
    
                start_city = np.random.randint(self.num_cities)
                ant.travel(start_city, self.pheromones, self.distances)
    
                if ant.distance > 0:
                    for i in range(len(ant.path) - 1):
                        pheromone_delta[ant.path[i]][ant.path[i + 1]] += 1 / ant.distance  # Δτij
    
                # Track best solution
                if ant.distance < best_distance:
                    best_distance = ant.distance
                    best_path = ant.path[:]
                    no_improvement = 0
                else:
                    no_improvement += 1
    
            # Update pheromones globally at the end of iteration
            self.pheromones = (self.pheromones * 0.9) + pheromone_delta  # τij(t+n) = ρ.τij(t) + Δτij
    
            if track_progress:
                distance_progress.append(best_distance)
            
            if no_improvement >= patience:
                print(f"No improvement for {patience}, stopping the algorithm")
                break
    
        if track_progress:
            return best_path, best_distance, distance_progress
        else:
            return best_path, best_distance
