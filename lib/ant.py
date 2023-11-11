import numpy as np
import random

from lib.classifier import fitness_function, random_selection

def useTACO(features, labels, ants=30, maxiter=50, alpha=1, beta=0.1, evaporation_rate=0.2, pheromone_value=1, desirability=1):
    dim = features.shape[0]
    pheromone_value = pheromone_value * np.ones([dim, dim]) #tau
    desirability = desirability * np.ones([dim, dim])       #eta
    goal = np.inf
    curve = np.inf
    fit = np.zeros((1, ants))
    t = 0
    while t <= maxiter:
        t += 1
        X = np.zeros((ants, dim))
        for i in range(ants):
            num_feat = np.random.randint(0, dim)
            X[i, 0] = np.random.randint(0, dim)
            k = []
            if num_feat > 0:
                for d in range(1, num_feat):
                    k.append(X[i,d-1])
                    rk = int(k[-1])-1
                    P = (pheromone_value[rk, :] ** alpha) * (desirability[rk, :] ** beta)
                    P[rk] = 0
                    prob = P / np.sum(P)
                    route = random_selection(prob)
                    X[i, d] = route
        X_bin = np.zeros((ants, dim))
        for i in range(ants):
            # Get the row 'X(i, :)'
            ind = X[i, :]
            # Remove elements equal to 0 from 'ind'
            ind = np.where(ind != 0)
            # Set corresponding elements in 'X_bin' to 1
            X_bin[i, ind] = 1
            print(X_bin)
        # for i in range(ants):
        #     fit[i] = fitness_function(features, labels, X_bin[i, 0])
        #     print(fit)

class AntColonyOptimization:
    def __init__(self, num_ants=30, num_iterations=50, alpha=1, beta=1, evaporation_rate=0.2, pheromone_deposit=0.1, distance_matrix=1):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.pheromone_matrix = np.ones((self.num_cities, self.num_cities))

    def run(self):
        best_path = None
        best_distance = float('inf')

        for iteration in range(self.num_iterations):
            ant_paths = self.generate_ant_paths()
            self.update_pheromones(ant_paths)
            
            for path, distance in ant_paths:
                if distance < best_distance:
                    best_path = path
                    best_distance = distance

            self.evaporate_pheromones()

        return best_path, best_distance

    def generate_ant_paths(self):
        ant_paths = []
        for ant in range(self.num_ants):
            current_city = random.randint(0, self.num_cities - 1)
            visited_cities = [current_city]
            path_distance = 0

            for _ in range(self.num_cities - 1):
                next_city = self.select_next_city(current_city, visited_cities)
                visited_cities.append(next_city)
                path_distance += self.distance_matrix[current_city][next_city]
                current_city = next_city

            path_distance += self.distance_matrix[current_city][visited_cities[0]]
            ant_paths.append((visited_cities, path_distance))

        return ant_paths

    def select_next_city(self, current_city, visited_cities):
        unvisited_cities = [city for city in range(self.num_cities) if city not in visited_cities]

        probabilities = []
        for city in unvisited_cities:
            pheromone = self.pheromone_matrix[current_city][city]
            distance = self.distance_matrix[current_city][city]
            probabilities.append((city, (pheromone ** self.alpha) * ((1 / distance) ** self.beta)))

        total_probability = sum(prob for city, prob in probabilities)
        probabilities = [(city, prob / total_probability) for city, prob in probabilities]

        selected_city = random.choices(population=[city for city, _ in probabilities], weights=[prob for _, prob in probabilities])[0]

        return selected_city

    def update_pheromones(self, ant_paths):
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    pheromone_change = 0
                    for path, distance in ant_paths:
                        if j in path and i in path:
                            pheromone_change += self.pheromone_deposit / distance
                    self.pheromone_matrix[i][j] = (1 - self.evaporation_rate) * self.pheromone_matrix[i][j] + pheromone_change

    def evaporate_pheromones(self):
        self.pheromone_matrix = (1 - self.evaporation_rate) * self.pheromone_matrix

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define the Ant Colony Optimization function
def ant_colony_optimization(feat, label, ants=30, iterations=10, tau=1, eta=1, alpha=1, beta=0.1, rho=0.2):
    # Parameters
    tau = 1
    eta = 1
    alpha = 1
    beta = 0.1
    rho = 0.2
    N = ants
    max_Iter = iterations
    # Objective function
    fun = fitness_function
    dim = feat.shape[1]

    tau = tau * np.ones((dim, dim))
    eta = eta * np.ones((dim, dim))
    fitG = np.inf
    fit = np.zeros(N)

    curve = np.inf
    t = 1

    while t <= max_Iter:
        X = np.zeros((N, dim))

        for i in range(N):
            num_feat = np.random.randint(1, dim + 1)
            X[i, 0] = np.random.randint(1, dim + 1)
            k = []
            ind = []

            if num_feat > 1:
                for d in range(1, num_feat):
                    k = k + [int(X[i, d - 1])]
                    P = (tau[k[-1]-1, :] ** alpha) * (eta[k[-1]-1, :] ** beta)
                    P[k[-1]-1] = 0
                    prob = P / np.sum(P)
                    route = random_selection(prob)
                    X[i, d] = route

            X_bin = np.zeros((N, dim))
            for i in range(N):
                ind = np.array(ind, dtype=int)  # Cast ind to an array of integers
                X_bin[i, ind[ind != 0] - 1] = 1  # Adjusted index

            for i in range(N):
                fit[i] = fun(feat, label, X_bin[i, :])

                if fit[i] < fitG:
                    Xgb = X[i, :]
                    fitG = fit[i]

            tauK = np.zeros((dim, dim))

            for i in range(N):
                tour = X[i, :]
                tour = tour[tour != 0]
                len_x = len(tour)
                tour = np.append(tour, tour[0])

                for d in range(len_x):
                    x = tour[d]
                    y = tour[d + 1]
                    tauK[x - 1, y - 1] += 1 / (1 + fit[i])

            tauG = np.zeros((dim, dim))
            tour = Xgb
            tour = tour[tour != 0]
            len_g = len(tour)
            tour = np.append(tour, tour[0])

            for d in range(len_g):
                x = tour[d]
                y = tour[d + 1]
                tauG[x - 1, y - 1] = 1 / (1 + fitG)

            tau = (1 - rho) * tau + tauK + tauG

            curve[t - 1] = fitG
            print('Iteration %d Best (ACO) = %f' % (t, curve[t - 1]))
            t += 1

    Sf = Xgb
    Sf = Sf[Sf != 0]
    sFeat = feat[:, Sf - 1]

    ACO = {
        'sf': Sf,
        'ff': sFeat,
        'nf': len(Sf),
        'c': curve,
        'f': feat,
        'l': label
    }

    return ACO

