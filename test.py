import os
from joblib import Parallel, delayed
import numpy as np
import numpy.typing as npt

from typing import List

from lib.aco import AntColonyOptimization

n = 25
ants = 5
iterations = 10
tau = np.random.uniform(0.01, 0.1, size=(n, n))   # Create a Pheromone Matrix representing Edges
eta = np.random.rand(n, n)   # Create a Heuristic Matrix representing value of Edges

tau = np.ones((n, n))
eta = np.ones((n, n))

rho = 0.2
alpha = 1
beta = 1

features = np.random.randint(0, n, (10, n))
labels = np.random.randint(0, 3, (10, ))

def transition(current: int, visited: List, tau: npt.NDArray, eta: npt.NDArray) -> npt.NDArray:
       M = N = (tau[current] ** alpha) * (eta[current] ** beta)
       M[visited] = 0 # Visited Nodes is set to 0 Probability
       M[current] = 0 # Current Node is set to 0 Probability
       P = N / M.sum()
       return np.random.choice(len(P), p=P)

def fitness_function(nodes):
       return np.random.rand()

def update_pheromone(tau, rho, delta_tau):
       return (1 - rho) * tau + delta_tau

def ant_tour(ant, tau, eta):
       node = np.random.choice(features)
       solution = {node}

       while len(solution) < len(features)//2:
              node = transition(node, list(solution), tau, eta)
              solution.add(node)
       
       fitness = fitness_function(solution)
       solution = np.array(list(solution))

       return solution, fitness

def explore(tau, eta, rho):
       global_best_solution = None
       global_best_accuracy = 0.0

       cores = os.cpu_count() // 2
       cores = cores if cores // 2 >= os.cpu_count() else cores + 1
       cores = 1

       for iteration in range(iterations):
              print(iteration)
              # Use Parralelization for each ant
              # Find the best local solutions for all ants
              local_best_solutions = Parallel(n_jobs=cores)(delayed(ant_tour)(ant, tau, eta) for ant in range(ants))

              # Initialize delta tau for pheromone update
              delta_tau = np.zeros_like(tau)

              for solution, accuracy in local_best_solutions:
                     for node in range(len(solution)-1):
                            delta_tau[solution[node], solution[node+1]] = 1 / accuracy
                            delta_tau[solution[node+1], solution[node]] = 1 / accuracy

                     if accuracy > global_best_accuracy:
                            global_best_solution = solution
                            global_best_accuracy = accuracy

              tau = update_pheromone(tau, rho, delta_tau)
              # print(np.round(tau))

       return global_best_solution, global_best_accuracy

