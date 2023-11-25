import os
import joblib
import numpy as np
import numpy.typing as npt

from typing import List
from lib.classifier import useSessionWrapperACO
from utilities.const import *

from lib.WrapperACO import WrapperACO
from utilities.util import predictImage


n = 5
ants = 5
iterations = 5
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
       N = (tau[current] ** 1) * (eta[current] ** 1)
       M = N
       M[visited] = 0 # Visited Nodes is set to 0 Probability
       M[current] = 0 # Current Node is set to 0 Probability
       P = N / M.sum()
       node = np.random.choice(len(P), p=P)
       print(f"\t", current, np.round(P,2))
       return node

def fitness_function(nodes):
       return np.random.rand()

def update_delta_tau(solution: List, quality: float, delta_tau: npt.NDArray):
       for node in range(len(solution)-1):
              # Update edge(i, j) delta tau based on quality of solution
              delta_tau[solution[node]][solution[node+1]] = 1 / (1 - quality + 1) 
       return delta_tau

def update_pheromone(tau, rho, delta_tau):
       return (1 - rho) * tau + delta_tau

def tour(ant, tau, eta, features, labels, subset_amount=0):
       nodes = features.shape[1]

       node = np.random.randint(nodes) # Select a random node
       subset_amount = nodes//2 if subset_amount < 1 else subset_amount
       path = [node] # Start with arbitrary node)

       while len(path) < subset_amount:
              node = transition(node, path, tau, eta)
              path.append(node)
       
       solution = np.array(path)
       fitness = fitness_function(features, labels, solution)
       print(f"\t\t", ant+1, solution, fitness)

       return solution, fitness

def explore(features, labels):
       global_accuracy = 0.0
       global_solution = None

       nodes = features.shape[1]
       tau = np.ones((nodes, nodes))
       eta = np.ones((nodes, nodes))

       for iteration in range(iterations):
              print(f"----------Iteration {iteration+1}---------")

              subset_amount = np.random.randint(2, nodes)
              local_solutions = []
              delta_tau = np.zeros_like(tau)

              for ant in range(ants):
                     local_solution, local_accuracy = tour(ant, tau, eta, features, labels, subset_amount)
                     local_solutions.append((local_solution, local_accuracy))

              for solution, accuracy in local_solutions:
                     delta_tau = update_delta_tau(solution, accuracy, delta_tau)
                     if accuracy > global_accuracy:
                            global_accuracy = accuracy
                            global_solution = solution

              tau = update_pheromone(tau, rho, delta_tau)

       print(f"Solution at current Iteration {iteration + 1} {global_solution} {global_accuracy:02f} {len(global_solution)}")
              
       return global_solution, global_accuracy
# print("Loading Features")
# X = selected_feature_indices = np.load(f"{DATA_PATH}/features.npy")
# Y = selected_feature_indices = np.load(f"{DATA_PATH}/labels.npy")
# print("Features Loaded")

# scaler.fit(X)
# X = scaler.transform(X)
# Y = label_encoder.fit_transform(Y)

# fit = lambda subset: fitness(X, Y, subset)

# aco = WrapperACO(fit, features.shape[1], ants=3, iterations=5, debug=1)
# useSessionWrapperACO(aco, fit, 1, 0, X, Y)
# useSessionWrapperACO(aco, fit, 1, 1, X, Y)
# useSessionWrapperACO(aco, fit, 1, 1, X, Y)
# useSessionWrapperACO(aco, fit, 1, 2, X, Y)


prediction = predictImage(r"dataset\google\hlt2.jpg", Model.BaseModel)
print(DISEASES, f'\n', np.round(prediction, 2))
print(DISEASES[np.argmax(prediction[0])])