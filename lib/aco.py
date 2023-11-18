import os
import joblib
import numpy as np
import numpy.typing as npt

from typing import List
from joblib import Parallel, delayed
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from utilities.const import *
np.set_printoptions(threshold=10)

class WrapperAntColonyOptimization:
    def __init__(self, model, ants=30, iterations=100, alpha=1.0, beta=1.0, rho=0.1, Q=1.0, debug=False, folds=1, parrallelization=False, cores=0):
        """
        Initialize the ACOFeatureSelection class with parameters for ACO-based feature selection.

        Parameters:
        - model : Model Evaluator
        - ants (int): Number of ants in the ACO colony.
        - iterations (int): Maximum number of iterations or generations.
        - alpha (float): Pheromone influence factor for the probability calculation.
        - beta (float): Heuristic influence factor for the probability calculation.
        - rho (float): Pheromone evaporation rate (between 0 and 1).
        - Q (float): Pheromone deposit amount for selected features.
        - delta (float): Pheromone deposit decay factor for selected features.
        """
        self.ants = ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

        self.model = model
        self.debug = debug
        self.folds = folds
        self.parallelization = parrallelization

        if cores == 0:
            self.cores = os.cpu_count() // 2
            self.cores = self.cores if self.cores // 2 >= os.cpu_count() else self.cores + 1 
        else:
            self.cores = cores
        assert self.cores <= os.cpu_count()

        if self.debug:
            print(f"Settings:\nalpha={self.alpha}\nbeta={self.beta}\nrho={self.rho}\nQ={self.Q}\nants={self.ants} iterations={self.iterations} folds={self.folds}")
            if self.parallelization:
                print(f"Parralelization={self.parallelization} cores={self.cores}")

    # Calculate transition probability and select a node based on the probability
    def transition(self, current: int, visited_nodes: List, tau: npt.NDArray, eta: npt.NDArray) -> int:
        N = (tau[current] ** self.alpha) * (eta[current] ** self.beta)
        M = N
        M[visited_nodes] = 0 # Visited Nodes is set to 0 Probability
        M[current] = 0 # Current Node is set to 0 Probability
        P = N / M.sum()
        node = np.random.choice(len(P), p=P)
        return node

    # Update delta tau for each path
    def delta_tau(self, solution: List, quality: float, delta_tau: npt.NDArray):
        for node in range(len(solution)-1):
            # Update edge(i, j) delta tau based on quality of solution for this path
            delta_tau[solution[node]][solution[node+1]] = self.Q / (1 - quality + 1) 
        return delta_tau
    
    # Update tau pheromone by applying evaporation and adding delta_tau to each tau edge explored
    def update_pheromone(self, tau: npt.NDArray, delta_tau: npt.NDArray):
        return (1 - self.rho) * tau + delta_tau
    
    # Objective function for wrapper feature selection, gets the path (solution subset)
    def fitness(self, features, labels, path):
        selected_features = features[:, path]

        if self.folds > 1:
            kfold = KFold(n_splits=self.folds, shuffle=True, random_state=42)
            scores = cross_val_score(self.model, selected_features, labels, cv=kfold)
            accuracy = scores.mean()
        else:
            X_train, X_test, y_train, y_test = train_test_split(selected_features, labels, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

        return accuracy
    
    # Ants explore the graph
    def tour(self, ant, tau, eta, features, labels, subset_amount=0):
        nodes = features.shape[1]
        assert subset_amount <= nodes

        node = np.random.randint(nodes) # Select a random node
        subset_amount = nodes//2 if subset_amount < 1 else subset_amount
        path = [node] # Start with arbitrary node

        while len(path) < subset_amount:
            node = self.transition(node, path, tau, eta) # Transitions from current node 'i' to the next node 'j'
            path.append(node) # Append the transitioned node
        
        solution = np.array(path) # Get the solution as an numpy array
        fitness = self.fitness(features, labels, solution) # Evaluate subset solution

        if self.debug >= 5:
            print(f"{ant+1}: {solution} {fitness*100:.2f} {len(solution)}")

        return solution, fitness

    # Apply ACO as feature selector
    def optimize(self, features, labels):
        global_accuracy = 0.0
        global_solution = None

        nodes = features.shape[1] # Get amount of features
        tau = np.ones((nodes, nodes)) # Initial pheromones
        eta = np.ones((nodes, nodes)) # Initial heuristics

        for iteration in range(self.iterations):
            if self.debug:
                print(f"Iteration {iteration+1}", end=" ")

            subset_amount = np.random.randint(2, nodes) # Initialize a random number of features
            local_solutions = [] # Store local solutions
            delta_tau = np.zeros_like(tau) # Initialize initial tau values

            if self.parallelization:
                local_solutions = Parallel(n_jobs=self.cores)(delayed(self.tour)(ant, tau, eta, features, labels, subset_amount) for ant in range(self.ants))
            else:
                for ant in range(self.ants):
                    local_solution, local_accuracy = self.tour(ant, tau, eta, features, labels, subset_amount)
                    local_solutions.append((local_solution, local_accuracy))

            for solution, accuracy in local_solutions:
                delta_tau = self.delta_tau(solution, accuracy, delta_tau) # Update delta tau for path(solution) found
                if accuracy > global_accuracy: # Obtain best local solution, if better local solution is the global solution
                    global_accuracy = accuracy
                    global_solution = solution

            tau = self.update_pheromone(tau, delta_tau)

            if self.debug:
                print(f"Solution:\t{iteration + 1} {global_solution} {global_accuracy:02f} {len(global_solution)}")
                
        return global_solution, global_accuracy

def WrapperAACO(features, labels, ants=30, iterations=100, parallel=False, debug=1, fold=1):
    label_encoder = LabelEncoder()
    numerical_labels = label_encoder.fit_transform(labels)
    
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(features)

    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, f"{SCALER_PATH}/AntColony.pkl")

    X = scaler.transform(X)

    print("Starting Ant Colony Optimization")
    aco = WrapperAntColonyOptimization(MODEL, ants=ants, iterations=iterations, debug=debug, folds=fold, parrallelization=parallel)
    selected_features, feature_fitness = aco.optimize(X, numerical_labels)
    print("Optimization with Ant Colony Complete")
    print(f"Solution: {np.sort(selected_features)} with {100*feature_fitness:.2f}% accuracy")

    np.save(f"{FEATURE_PATH}/AntColony.npy", selected_features)

    X_train, X_test, Y_train, Y_test = train_test_split(X[:, selected_features], numerical_labels, test_size=0.2, random_state=42)

    svm = MODEL
    svm.fit(X_train, Y_train)

    Y_pred = svm.predict(X_test)
    predicted_class_labels = label_encoder.inverse_transform(Y_pred)

    report = classification_report(Y_test, Y_pred, target_names=label_encoder.classes_, zero_division='warn')
    overall_accuracy = accuracy_score(Y_test, Y_pred)

    print("Test ACO Classification Report:")
    print(report)
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
    print(f"Features: {X.shape[1]} & {X[:, selected_features].shape[1]}")

    return svm, overall_accuracy
