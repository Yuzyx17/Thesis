from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from pyswarm import pso
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np

def fitness_function(feat, label, selected_features, opts):
    # Get the selected features
    selected_feat = feat[:, selected_features]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(selected_feat, label, test_size=0.2, random_state=42)

    # Create and train the MSVM classifier
    clf = SVC()
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy (you can use other metrics as needed)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

# Roulette Wheel Selection
def random_selection(prob):
    # Cumulative summation
    C = np.cumsum(prob)
    # Random one value, most probability value [0~1]
    P = np.random.rand()
    # Roulette wheel
    for i in range(len(C)):
        if C[i] > P:
            return i + 1

def useBase(features, labels):
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
    svm = SVC(kernel='rbf')

    # Train the model on the training data
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

    # Print overall accuracy
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")


def usePSO(features, labels):
    # Convert class labels to numerical labels
    label_encoder = LabelEncoder()
    numerical_labels = label_encoder.fit_transform(labels)

    # Define the fitness function for PSO
    def fitness_function(feature_subset):
        # Extract selected features
        selected_features = np.where(feature_subset > 0.5)[0]

        # Split the data into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(features[:, selected_features], numerical_labels, test_size=0.2, random_state=42)

        # Create an MSVM model with an RBF kernel
        svm = SVC(kernel='rbf')

        # Train the model on the training data
        svm.fit(X_train, Y_train)

        # Make predictions on the test set
        Y_pred = svm.predict(X_test)

        # Calculate and return the negative accuracy (to maximize)
        return -accuracy_score(Y_test, Y_pred)

    # Define the bounds for each feature (0 for unselected, 1 for selected)
    n_features = features.shape[1]
    lb = np.zeros(n_features)
    ub = np.ones(n_features)

    # Run PSO to select the optimal feature subset
    best_feature_subset, _ = pso(fitness_function, lb, ub)

    # Extract the selected features
    selected_features = np.where(best_feature_subset > 0.5)[0]

    # Split the data into training and testing sets using the selected features
    X_train, X_test, Y_train, Y_test = train_test_split(features[:, selected_features], numerical_labels, test_size=0.2, random_state=42)

    # Create an SVM classifier
    svm = SVC(kernel='rbf')

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
    print("Classification Report:")
    print(report)

    # Print overall accuracy
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")

def useACO(features, labels):
    # Convert class labels to numerical labels
    label_encoder = LabelEncoder()
    numerical_labels = label_encoder.fit_transform(labels)

    # Parameters for ACO
    n_ants = 100
    n_iterations = 100
    n_features = features.shape[1]
    pheromone_0 = 1.0 / (n_features * n_ants)

    # Initialize pheromone trails
    pheromone = np.full((n_features,), pheromone_0)

    # Run ACO
    for it in range(n_iterations):
        # Generate solutions for all ants
        for ant in range(n_ants):
            # Binary array for selected features
            feature_subset = np.zeros(n_features, dtype=bool)

            # Select features based on pheromone trails
            for feature in range(n_features):
                if np.random.rand() < pheromone[feature]:
                    feature_subset[feature] = True

            # Calculate fitness of the solution
            X_train, X_test, Y_train, Y_test = train_test_split(features[:, feature_subset], numerical_labels, test_size=0.2, random_state=42)
            svm = SVC(kernel='rbf')
            svm.fit(X_train, Y_train)
            Y_pred = svm.predict(X_test)
            fitness = accuracy_score(Y_test, Y_pred)

            # Update pheromone trails
            pheromone[feature_subset] *= (1.0 + fitness)
            pheromone[~feature_subset] *= (1.0 - fitness)

    # Extract the selected features
    selected_features = np.where(pheromone > np.median(pheromone))[0]

    # Split the data into training and testing sets using the selected features
    X_train, X_test, Y_train, Y_test = train_test_split(features[:, selected_features], numerical_labels, test_size=0.2, random_state=42)

    # Create an SVM classifier
    svm = SVC(kernel='rbf')

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
    print("Classification Report:")
    print(report)

    # Print overall accuracy
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
