import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from pyswarm import pso
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm
import numpy as np
from lib.WrapperACO import WrapperACO

from utilities.const import *

def BaseModel(features, labels):
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Create an MSVM model with an RBF kernel
    svm = MODEL

    # Train the model on the training data
    svm.fit(X_train, Y_train)

    # Make predictions on the test set
    Y_pred = svm.predict(X_test)

    # Generate a classification report
    report = classification_report(Y_test, Y_pred, target_names=label_encoder.classes_, zero_division='warn')

    # Calculate the overall accuracy
    overall_accuracy = accuracy_score(Y_test, Y_pred)

    # Print the classification report with class-wise accuracy
    print("Classification Report:")
    print(report)

    # Print overall accuracy
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
    print(f"Features: {features.shape[0]}")
    
    print(label_encoder.classes_)
    return svm, overall_accuracy, None

def useBaseCV(features, labels, cv=5):
    # Convert class labels to numerical labels
    unique_labels = np.unique(labels)
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    numerical_labels = np.array([label_to_id[label] for label in labels])

    # Create a SimpleImputer to handle missing values (replace 'mean' with your preferred strategy)
    imputer = SimpleImputer(strategy='mean')

    # Apply imputation to your feature data
    X_imputed = imputer.fit_transform(features)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_imputed, numerical_labels, test_size=0.2, random_state=42)

    # Create an MSVM model with an RBF kernel
    svm = MODEL

    # Perform cross-validation and obtain scores
    cv_scores = cross_val_score(svm, X_imputed, numerical_labels, cv=cv)

    # Print cross-validation scores
    print("Cross-Validation Scores:", cv_scores)
    print("Mean CV Score:", cv_scores.mean())

    # Fit the model on the entire training data
    svm.fit(X_train, Y_train)

    # Make predictions on the test set
    Y_pred = svm.predict(X_test)

    # Convert numerical labels back to original class labels
    predicted_class_labels = [unique_labels[label] for label in Y_pred]

    # Generate a classification report
    report = classification_report(Y_test, Y_pred, target_names=unique_labels, zero_division='warn')

    # Calculate the overall accuracy
    overall_accuracy = accuracy_score(Y_test, Y_pred)

    # Print the classification report with class-wise accuracy
    print("Classification Report:")
    print(report)
    print(f"Features: {features.shape[0]}")

    return svm

def usePSO(features, labels, swarm=30, iterations=100):
    # Convert class labels to numerical labels
    label_encoder = LabelEncoder()
    numerical_labels = label_encoder.fit_transform(labels)
    
    imputer = SimpleImputer(strategy='mean')
    # Apply imputation to your feature data
    X = imputer.fit_transform(features)
    # Initialize the scaler
    scaler = StandardScaler()

    # Fit on the imputed data
    scaler.fit(X)
    joblib.dump(scaler, f"{SCALER_PATH}/pso.pkl")
    X = scaler.transform(X)

    # Define the fitness function for PSO
    def fitness_function(feature_subset):
        # Extract selected features
        selected_features = np.where(feature_subset > 0.5)[0]

        # Split the data into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X[:, selected_features], numerical_labels, test_size=0.2, random_state=42)

        # Create an MSVM model with an RBF kernel
        svm = SVC(C=10, kernel='rbf', probability=True)

        # Train the model on the training data
        svm.fit(X_train, Y_train)

        # Make predictions on the test set
        Y_pred = svm.predict(X_test)

        # Calculate and return the negative accuracy (to maximize)
        return -accuracy_score(Y_test, Y_pred)
    

    # Define the bounds for each feature (0 for unselected, 1 for selected)
    n_features = X.shape[1]
    lb = np.zeros(n_features)
    ub = np.ones(n_features)

    print("Starting PSO")
    best_feature_subset, _ = pso(fitness_function, lb, ub, swarmsize=swarm, maxiter=iterations, debug=True)
    # Convert binary array to indices
    selected_features = np.where(best_feature_subset > 0.6)[0]
    print("Optimization with PSO Complete")
    # Save the indices of the selected features
    np.save(f"{FEATURE_PATH}/pso.npy", selected_features)
    # Split the data into training and testing sets using the selected features
    X_train, X_test, Y_train, Y_test = train_test_split(X[:, selected_features], numerical_labels, test_size=0.2, random_state=42)

    # Create an SVM classifier
    svm = MODEL

    # Train the model on the training data
    svm.fit(X_train, Y_train)

    # Make predictions on the test set
    Y_pred = svm.predict(X_test)

    # Convert numerical labels back to original class labels
    predicted_class_labels = label_encoder.inverse_transform(Y_pred)

    # Generate a classification report
    report = classification_report(Y_test, Y_pred, target_names=label_encoder.classes_, zero_division='warn')

    # Calculate the overall accuracy
    overall_accuracy = accuracy_score(Y_test, Y_pred)

    # Print the classification report with class-wise accuracy
    print("PSO Classification Report:")
    print(report)

    # Print overall accuracy
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
    print(f"Features: {X.shape[0]} & {X[:, selected_features].shape[0]}")

    return svm

def useACO(features, labels, ants=30, iterations=100, alpha=1.0, beta=1.0, rho=0.2, Q=1.0, delta=0.1):
    
    # def calculate_transition_probabilities(tau, eta, alpha, beta, visited_features):
    #     probabilities = (tau ** alpha) * (eta ** beta)
    #     probabilities[list(visited_features)] = 0  # Set probabilities of visited features to 0
    #     probabilities /= probabilities.sum()
    #     return probabilities

    # def select_next_feature(probabilities):
    #     return np.random.choice(len(probabilities), p=probabilities)

    # def evaluate_solution(features, labels, selected_features):
    #     selected_feat = features[:, selected_features]
    #     X_train, X_test, y_train, y_test = train_test_split(selected_feat, labels, test_size=0.2, random_state=42)
        
    #     clf = SVC(C=10, kernel='rbf', probability=True)
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_test)
    #     accuracy = accuracy_score(y_test, y_pred)
        
    #     return accuracy

    # def evaporate_pheromone(tau, rho):
    #     return (1 - rho) * tau

    # def deposit_pheromone(tau, solution, delta, Q):
    #     for feature in solution:
    #         tau[feature] += delta * Q

    # def abacoh_feature_selection(features, labels, m, Imax, alpha, beta, rho, Q, delta):
    #     n_ants = m
    #     n_features = features.shape[1]

    #     tau_min, tau_max = 0.01, 0.1
    #     tau = np.random.uniform(tau_min, tau_max, size=(n_features,))

    #     # Initialize heuristic information matrix eta based on F-score (not provided)
    #     eta = np.random.rand(n_features)

    #     global_best_solution = []
    #     global_best_accuracy = 0.0

    #     for iteration in range(Imax):
    #         local_best_solutions = []

            
    #         for ant in tqdm(range(n_ants)):
    #             visited_features = set()
    #             ant_solution = []

    #             while len(visited_features) < n_features:
    #                 probabilities = calculate_transition_probabilities(tau, eta, alpha, beta, visited_features)
    #                 next_feature = select_next_feature(probabilities)

    #                 visited_features.add(next_feature)
    #                 ant_solution.append(next_feature)

    #             accuracy = evaluate_solution(features, labels, ant_solution)
    #             local_best_solutions.append((ant_solution, accuracy))

    #         local_best_solutions.sort(key=lambda x: x[1], reverse=True)

    #         if local_best_solutions[0][1] > global_best_accuracy:
    #             global_best_solution = local_best_solutions[0][0]
    #             global_best_accuracy = local_best_solutions[0][1]
    #             print(f"New Best Solution at Iteration {iteration+1}: {X[:, global_best_solution]}")
    #         else:
    #             print(f"Best Solution at Iteration {iteration+1}: {X[:, global_best_solution]}")
    #         tau = evaporate_pheromone(tau, rho)
    #         deposit_pheromone(tau, global_best_solution, delta, Q)

    #     return global_best_solution
    ...

def useGridSVC(features, labels, param_grid, cv=2):
    # Convert class labels to numerical labels
    
    numerical_labels = label_encoder.fit_transform(labels)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(features, numerical_labels, test_size=0.2, random_state=42)

    # Create an SVM classifier
    svm = SVC()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=10)

    # Fit the grid search to the data
    grid_search.fit(X_train, Y_train)

    # Get the best parameters and estimator
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_

    # Use the best estimator to make predictions
    Y_pred = best_estimator.predict(X_test)

    # Convert numerical labels back to original class labels
    predicted_class_labels = label_encoder.inverse_transform(Y_pred)

    # Generate a classification report
    report = classification_report(Y_test, Y_pred, target_names=label_encoder.classes_, zero_division='warn')

    # Calculate the overall accuracy
    overall_accuracy = accuracy_score(Y_test, Y_pred)

    # Print the best parameters, overall accuracy, and classification report
    print("Best Parameters:", best_params)
    print("Best Estimators:", best_estimator)
    print("Overall Accuracy: {:.2f}%".format(overall_accuracy * 100))
    print("Classification Report:")
    print(report)

    return best_params, best_estimator

def createModel(features, labels, selectedFeatures=None):
    features = features[:, selectedFeatures] if selectedFeatures is not None else features
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=TEST_SIZE, random_state=R_STATE)

    svm = MODEL
    svm.fit(X_train, Y_train)

    Y_pred = svm.predict(X_test)
    report = classification_report(Y_test, Y_pred, target_names=label_encoder.classes_, zero_division='warn')
    overall_accuracy = accuracy_score(Y_test, Y_pred)

    print("Wrapper ACO Model Classification Report:")
    print(report)
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
    print(f"Features: {features.shape[1]} & {features[:,selectedFeatures].shape[1]}")

    return svm, overall_accuracy

def useWrapperACO(features, labels, aco):
    print("Starting Ant Colony Optimization")
    solution, quality = aco.optimize()
    print("Optimization with Ant Colony Complete")
    print(f"Solution: {np.sort(solution)} with {100*quality:.2f}% accuracy")

    model, accuracy = createModel(features, labels, solution)
    return model, accuracy, solution