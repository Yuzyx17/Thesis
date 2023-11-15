import os
import sys
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
from tqdm import tqdm

from utilities.util import progressBar

class ACOFeatureSelection:
    def __init__(self, n_ants=30, max_iterations=100, alpha=1.0, beta=1.0, rho=0.2, Q=1.0, delta=0.1, debug=True):
        """
        Initialize the ACOFeatureSelection class with parameters for ACO-based feature selection.

        Parameters:
        - n_ants (int): Number of ants in the ACO colony.
        - max_iterations (int): Maximum number of iterations or generations.
        - alpha (float): Pheromone influence factor for the probability calculation.
        - beta (float): Heuristic influence factor for the probability calculation.
        - rho (float): Pheromone evaporation rate (between 0 and 1).
        - Q (float): Pheromone deposit amount for selected features.
        - delta (float): Pheromone deposit decay factor for selected features.
        """
        self.n_ants = n_ants
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.delta = delta
        self.debug = debug

        assert rho >= 0 and rho <= 1

    def _calculate_transition_probabilities(self, tau, eta, visited_features):
        """
       Params:
       ------
       current: int
              The current node
       visited_nodes: List
              list of all visited nodes
       tau: NDArray
              Edge Matrix of Pheromone Levels
       eta: NDArray
              Edge Matrix of Heuristic Desirability
       ------
       Given formula:
       ------
       Pᵏᵢⱼ = N/S\n
       N = (τᵃᵢⱼ) * (ηᵇᵢⱼ)\n
       M = Σᵢ∈allowedₗ (τᵃᵢₗ) * (ηᵇᵢₗ) \n

       where:
              k = ants\n
              P = Probability\n
              (i, j) = All edges\n
              (i, l) = All available edges\n
              τ = tau
              η = eta
              α = alpha\n
              β = beta\n
        
        Gets the next node base on the probability
       ------
       Returns: 
       ------
              Next node
       """
        probabilities = (tau ** self.alpha) * (eta ** self.beta)
        probabilities[list(visited_features)] = 0
        probabilities /= probabilities.sum()
        return probabilities

    def _select_next_feature(self, probabilities):
        return np.random.choice(len(probabilities), p=probabilities)

    def _evaluate_solution(self, features, labels, selected_features):
        selected_feat = features[:, selected_features]
        X_train, X_test, y_train, y_test = train_test_split(selected_feat, labels, test_size=0.2, random_state=42)
        
        clf = SVC(C=10, kernel='rbf', probability=True)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy

    def _evaporate_pheromone(self, tau):
        tau = (1 - self.rho) * tau
        return np.clip(tau, 0.01, 0.1)

    def _deposit_pheromone(self, tau, solution):
        """
        Params:
        ------
        solution: List
                The current solution
        quality: float
                The quality of the solution
        tau: NDArray
                Edge Matrix of Pheromone Levels
        ------
        Given formula:
        ------
        Pᵏᵢⱼ = N/S\n
        N = (τᵃᵢⱼ) * (ηᵇᵢⱼ)\n
        M = Σᵢ∈allowedₗ (τᵃᵢₗ) * (ηᵇᵢₗ) \n

        where:
                k = ants\n
                P = Probability\n
                (i, j) = All edges\n
                (i, l) = All available edges\n
                τ = tau
                η = eta
                α = alpha\n
                β = beta\n
            
            Gets the next node base on the probability
        ------
        Returns: 
        ------
                Next node
        """
        for feature in solution:
            tau[feature] += self.delta * self.Q
        return tau

    # Mixed implementation
    def _calculate_significance(self, eta, solution, accuracy):
        return eta[solution] * accuracy
    
    def fit(self, features, labels):
        n_samples, n_features = features.shape
        tau_min, tau_max = 0.01, 0.1
        tau = np.random.uniform(tau_min, tau_max, size=(n_features,))
        eta = np.random.rand(n_features)

        global_best_solution = []
        global_best_accuracy = 0.0

        cores = os.cpu_count() // 2
        cores = cores if cores // 2 >= os.cpu_count() else cores + 1

        for iteration in range(self.max_iterations):
            if self.debug:
                print(f"Iteration: {iteration+1}")
            def search_solution(ant_id):
                # local_solution = []
                # local_accuracy = 0.0

                visited_features = set()
                ant_solution = []

                # # With Thresholding
                # while len(visited_features) < n_features:
                #     probabilities = self._calculate_transition_probabilities(tau, eta, visited_features)
                #     threshold = np.mean(probabilities) * 1.5 # Change

                #     next_feature = self._select_next_feature(probabilities)
                #     visited_features.add(next_feature)

                #     if probabilities[next_feature] > threshold: 
                #         ant_solution.append(next_feature)
                    
                # With Pre-determined subset amount
                while len(visited_features) < n_features//2:
                    probabilities = self._calculate_transition_probabilities(tau, eta, visited_features)
                    next_feature = self._select_next_feature(probabilities)
                    visited_features.add(next_feature)
                    ant_solution.append(next_feature)

                local_accuracy = self._evaluate_solution(features, labels, ant_solution)
                local_solution = np.array(ant_solution)
                # if accuracy > local_best_accuracy:
                #     local_best_solution = list(ant_solution)
                #     local_best_accuracy = accuracy
                return local_solution, local_accuracy

            # Parallelize the search for solutions
            local_best_solutions = Parallel(n_jobs=cores)(delayed(search_solution)(ant_id) for ant_id in range(self.n_ants))

            for solution, accuracy in local_best_solutions:
                if accuracy > global_best_accuracy:
                    global_best_solution = np.array(solution)
                    global_best_accuracy = accuracy

            print(f"Best Solution at Iteration {iteration+1}: {global_best_solution} {global_best_accuracy} {global_best_solution.shape}")

            tau = self._evaporate_pheromone(tau)
            tau = self._deposit_pheromone(tau, global_best_solution)

        self.selected_features = global_best_solution


    def transform(self, features):
        return features[:, self.selected_features]

    def fit_transform(self, features, labels):
        self.fit(features, labels)
        return self.selected_features
