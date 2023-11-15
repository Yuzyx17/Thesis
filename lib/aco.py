import os
import sys
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
from sklearn.svm import SVC
from utilities.const import *


class AntColonyOptimization:
    def __init__(self, ants=30, iterations=100, alpha=1.0, beta=1.0, rho=0.2, Q=1.0, debug=True):
        """
        Initialize the ACOFeatureSelection class with parameters for ACO-based feature selection.

        Parameters:
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

        self.debug = debug
        self.progress = 0

    # Calculate transition probability and select a node based on the probability
    def transition(self, current: int, visited_nodes: List, tau: npt.NDArray, eta: npt.NDArray) -> int:
        M = N = (tau[current] ** self.alpha) * (eta[current] ** self.beta)
        M[visited_nodes] = 0 # Visited Nodes is set to 0 Probability
        M[current] = 0 # Current Node is set to 0 Probability
        P = N / M.sum()
        node = np.random.choice(len(P), p=P)
        return node

    # Update delta tau for each ant
    def delta_tau(self, solution: List, quality: float, delta_tau: npt.NDArray):
        for node in range(len(solution)-1):
            delta_tau[solution[node]][solution[node+1]] = self.Q / (1 + quality)
        return delta_tau
    
    # Update pheromone
    def update_pheromone(self, tau: npt.NDArray, delta_tau: npt.NDArray):
        return (1 - self.rho) * tau + delta_tau
    
    # Use SVM as a fitness function wrapping the SVM with ACO
    def fitness_k(self, features, labels, path):
        selected_features = features[:, path]
        svm = SVC(C=10, kernel='rbf', probability=True)

        kfold = KFold(n_splits=FOLD, shuffle=True, random_state=42)
        scores = cross_val_score(svm, selected_features, labels, cv=kfold)
        accuracy = scores.mean()

        return accuracy
    
    def fitness(self, features, labels, path):
        selected_features = features[:, path]
        svm = SVC(C=10, kernel='rbf', probability=True)

        X_train, X_test, y_train, y_test = train_test_split(selected_features, labels, test_size=0.2, random_state=42)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    # Ants explore the graph
    def tour(self, ant, tau, eta, features, labels, subset_amount=0):
        nodes = features.shape[1]

        assert subset_amount <= nodes

        node = np.random.randint(nodes) # Select a random node
        subset_amount = nodes//2 if subset_amount < 1 else subset_amount
        path = [node] # Start with arbitrary node)

        while len(path) < subset_amount:
            next_node = self.transition(node, path, tau, eta)
            path.append(next_node)
        
        solution = np.array(path)
        fitness = self.fitness(features, labels, solution)

        if self.debug:
            self.progress += 1
            sys.stdout.write("Explored Ants: %d  \r" % (self.progress) )
            sys.stdout.flush()

        return solution, fitness

    # Apply ACO as feature selector
    def aco(self, features, labels):
        global_accuracy = 0.0
        global_solution = None

        nodes = features.shape[1]
        tau = np.ones((nodes, nodes))

        for iteration in range(self.iterations):
            if self.debug:
                print(f"Iteration {iteration+1}")
                self.progress = 0

            subset_amount = np.random.randint(1, nodes)
            local_solutions = []
            delta_tau = np.ones_like(tau)
            eta = np.random.rand(nodes, nodes)

            for ant in range(self.ants):
                local_solution, local_accuracy = self.tour(ant, tau, eta, features, labels, subset_amount)
                local_solutions.append((local_solution, local_accuracy))

            for solution, accuracy in local_solutions:
                delta_tau = self.delta_tau(solution, accuracy, delta_tau)
                if accuracy > global_accuracy:
                    global_accuracy = accuracy
                    global_solution = solution

            tau = self.update_pheromone(tau, delta_tau)
            if self.debug:
                print(f"Solution at current Iteration {iteration + 1} {global_solution} {global_accuracy:02f} {len(global_solution)}")
                
        return global_solution, global_accuracy
    
    # Apply ACO with Parallelization as feature selector
    def parallell_aco(self, features, labels):
        global_accuracy = 0.0
        global_solution = None

        nodes = features.shape[1]
        tau = np.ones((nodes, nodes))

        for iteration in range(self.iterations):
            if self.debug:
                print(f"Iteration {iteration+1}")

            subset_amount = np.random.randint(1, nodes)
            delta_tau = np.ones_like(tau)
            eta = np.random.rand(nodes, nodes)

            local_solutions = Parallel(n_jobs=CORES)(delayed(self.tour)(ant, tau, eta, features, labels, subset_amount) for ant in range(self.ants))

            for solution, accuracy in local_solutions:
                delta_tau = self.delta_tau(solution, accuracy, delta_tau)
                if accuracy > global_accuracy:
                    global_accuracy = accuracy
                    global_solution = solution

            tau = self.update_pheromone(tau, delta_tau)
            if self.debug:
                print(f"Solution at current Iteration {iteration + 1} {global_solution} {global_accuracy:02f} {len(global_solution)}")
                
        return global_solution, global_accuracy

def TestACO(features, labels, ants=30, iterations=100, parallel=False):
    label_encoder = LabelEncoder()
    numerical_labels = label_encoder.fit_transform(labels)
    
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(features)

    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, f"{SCALER_PATH}/test-aco.pkl")

    X = scaler.transform(X)

    print("Starting ACO")
    aco = AntColonyOptimization(ants=ants, iterations=iterations)
    selected_features, _ = aco.parallell_aco(X, numerical_labels) if parallel else aco.aco(X, numerical_labels)
    print("Optimization with ACO Complete")

    np.save(f"{FEATURE_PATH}/test-aco.npy", selected_features)

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

    return svm