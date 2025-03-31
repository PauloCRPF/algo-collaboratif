## imports ##
import numpy as np
import matplotlib.pyplot as plt
from ant_colony_tsp import Ant, Environment
from numpy.random import randint, binomial, choice

class Evolver:
    def __init__(self, n_population, positions, n_cities, alpha_vals, beta_vals, rho_vals, n_generations, n_iter_env, p_keep, p_cross, p_mutate):
        #### optimizer for enviroment ####
        # n_population: numer of environments
        # positions: [(a,b):a,b cities]
        # n_cities: number of cities
        # alpha_vals: list of alpha values
        # beta_vals: list of beta values
        # rho_vals: list of rho values
        # n_generations: number of generations
        # n_iter_env: number of iterations for each environment
        # p_keep: percentage of environments to keep
        # p_cross: percentage of environments to cross
        # p_mutate: percentage of environments to mutate
        #####
        self.n_population = n_population
        self.positions = positions
        self.n_cities = n_cities
        self.alpha_vals = alpha_vals
        self.beta_vals = beta_vals
        self.rho_vals = rho_vals
        self.n_generations = n_generations
        self.n_iter_env = n_iter_env
        self.p_keep = p_keep
        self.p_cross = p_cross
        self.p_mutate = p_mutate
        #
        self.n_keep = int(self.p_keep * self.n_population)
        self.n_cross = int(self.p_cross * self.n_population)
        self.n_mutate = int(self.p_mutate * self.n_population)
        #
        self.evolved = False
        self.best = None
        # santity check
        self.sanity_check()
        # outputs
        self.out_distances = []
        self.out_paths = []
    
    def __str__(self):
        if self.evolved:
            best = self.best
            best_genes = [best.alpha, best.beta, best.gamma]
            return f'Best environment: {best_genes}'
        else:
            return 'Evolver: not evolved'

    def sanity_check(self):
        assert self.n_population > 0
        assert self.p_keep + self.p_cross + self.p_mutate < 1
        assert self.p_cross* self.n_population > 1
        assert self.p_mutate* self.n_population > 1
        assert self.p_keep * self.n_population > 1
        assert len(self.positions) == self.n_cities
        assert len(self.alpha_vals) > 0
        assert len(self.beta_vals) > 0
        assert len(self.rho_vals) > 0
        assert self.n_generations > 0
        assert self.n_iter_env > 0

    def evolve(self):  
        # create population
        population = self.create_population(self.n_population)
        # sort
        population, paths, distances = self.sort_population(population)
        print(paths[0])
        print(distances[0])
        print("---")
        # loop
        for i in range(self.n_generations):
            ## change by group
            # keep
            if i < self.n_keep:
                continue
            # cross
            elif i < self.n_keep + self.n_cross:
                parent = population[i]
                child = self.cross_population(population, parent)
                population[i] = child
            # mutate
            elif  i < self.n_keep + self.n_cross + self.n_mutate:
                old = population[i]
                new = self.mutate(old)
                population[i] = new
            # new
            else:
                # number of new
                n_new = self.n_population - self.n_keep - self.n_cross - self.n_mutate
                # create new
                new = self.create_population(n_new)
                population[-n_new:] = new

            ## sort ##
            population, paths, distances = self.sort_population(population)
            self.out_distances.append(distances[0])
            self.out_paths.append(paths[0])
            ## print ##
            print(f'iteration: {i} out of {self.n_generations}, distance: {distances[0]}')
        # set best
        self.evolved = True
        self.best = population[0]
        # return
        return population[0]  

    def simulate_env(self,env):
        path, distance = env.simulate(self.n_iter_env)
        return path, distance
    
    def create_population(self, n_population):
        population = [Environment(
            num_cities = self.n_cities, 
            alpha = choice(self.alpha_vals),
            beta = choice(self.beta_vals),
            gamma = choice(self.rho_vals),
            num_ants = self.n_cities,
            positions = self.positions
        ) for _ in range(n_population)]
        return population
    
    def sort_population(self, population):
        paths = []
        distances = []
        for env in population:
            path, distance = self.simulate_env(env)
            paths.append(path)
            distances.append(distance)
        idx = np.argsort(distances)
        return [population[i] for i in idx], [paths[i] for i in idx], [distances[i] for i in idx]
    
    def cross_population(self, population, parent):
        # parent best
        idx_best = randint(0, self.n_keep)
        parent_best = population[idx_best]
        genes_best = [parent_best.alpha, parent_best.beta, parent_best.gamma]
        # other parent
        other_parent = parent
        genes_other = [other_parent.alpha, other_parent.beta, other_parent.gamma]
        # choose genes from best
        which_genes = binomial(1, 0.5, size = 3)     
        # cross genes
        child_genes = [genes_best[i] if which_genes[i] == 1 else genes_other[i] for i in range(3)]
        # create child
        child = Environment(
            num_cities = self.n_cities, 
            alpha = child_genes[0],
            beta = child_genes[1],
            gamma = child_genes[2],
            num_ants = self.n_cities,
            positions = self.positions
        )
        return child
    
    def mutate(self, old):
        # which gene to mutate
        which_gene = randint(0, 3)
        # mutate gene
        if which_gene == 0:
            old.alpha = choice(self.alpha_vals)
        elif which_gene == 1:
            old.beta = choice(self.beta_vals)
        else:
            old.gamma = choice(self.rho_vals)
        # return
        return old

def run_evolution(points):
    from evolution import Evolver

    n_cities = len(points)
    positions = {i: points[i] for i in range(n_cities)}
    
    evolver = Evolver(
        n_population=20,                  # ✅ Aumente a população
        positions=positions,
        n_cities=n_cities,
        alpha_vals=[0.5, 1, 2, 3],
        beta_vals=[2, 3, 4, 5],
        rho_vals=[0.05, 0.1, 0.2],
        n_generations=10,
        n_iter_env=100,
        p_keep=0.2,                       # ✅ Garante pelo menos 2 mantidos (0.2 * 20 = 4)
        p_cross=0.4,
        p_mutate=0.3                      # ✅ Soma das três deve ser < 1
    )

    best_env = evolver.evolve()
    return best_env.alpha, best_env.beta, best_env.gamma
