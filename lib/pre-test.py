import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

class ABACOFeatureSelector:
    def __init__(self, m=10, Imax=100, alpha=1.0, beta=2.0, rho=0.1, min_pheromone=0.1, max_pheromone=1.0):
        self.m = m
        self.Imax = Imax
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.min_pheromone = min_pheromone
        self.max_pheromone = max_pheromone
        self.n_features = None
        self.pheromone = None
        self.heuristic_info = None
        self.best_solution = None
        self.best_accuracy = 0.0

    def _initialize(self, X):
        self.n_features = X.shape[1]
        self.pheromone = np.ones((self.n_features, self.n_features)) * self.min_pheromone
        self.heuristic_info = np.random.rand(self.n_features, self.n_features)

    def _evaluate_subset(self, X, y, subset):
        selected_features = np.where(subset == 1)[0]
        X_subset = X[:, selected_features]
        classifier = SVC()
        scores = cross_val_score(classifier, X_subset, y, cv=5)  # Use cross-validation for evaluation
        return scores.mean()

    def _transition_rule(self, current_feature):
        probabilities = (self.pheromone[current_feature] ** self.alpha) * (self.heuristic_info[current_feature] ** self.beta)
        probabilities[current_feature] = 0  # Exclude the current feature
        probabilities /= probabilities.sum()
        next_feature = np.random.choice(range(self.n_features), p=probabilities)
        return next_feature

    def fit(self, X, y):
        self._initialize(X)
        
        for iteration in range(self.Imax):
            for ant in range(self.m):
                current_feature = np.random.randint(self.n_features)
                ant_solution = np.zeros(self.n_features, dtype=int)
                ant_solution[current_feature] = 1

                for _ in range(self.n_features - 1):
                    next_feature = self._transition_rule(current_feature)
                    ant_solution[next_feature] = 1
                    current_feature = next_feature

                accuracy = self._evaluate_subset(X, y, ant_solution)

                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_solution = ant_solution.copy()

            self.pheromone = (1 - self.rho) * self.pheromone
            best_ant = np.argmax([self._evaluate_subset(X, y, self.best_solution) for _ in range(self.m)])
            
            for feature in range(self.n_features):
                if self.best_solution[feature] == 1:
                    self.pheromone[feature, :] += 1.0 / self.best_accuracy

    def transform(self, X):
        global_best_subset = np.where(self.best_solution == 1)[0]
        final_X_subset = X[:, global_best_subset]
        return final_X_subset
