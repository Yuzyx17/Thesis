import json
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
from lib.WrapperPSO import WrapperPSO

from utilities.const import *
from utilities.util import saveModel

def BaseModel(features, labels):
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Create an MSVM model with an RBF kernel
    svm = CLASSIFIER

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
    svm = CLASSIFIER

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
    svm = CLASSIFIER

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
    X_train, X_test = features
    Y_train, Y_test = labels

    X_train = X_train[:, selectedFeatures] if selectedFeatures is not None else X_train
    X_test = X_test[:, selectedFeatures] if selectedFeatures is not None else X_test

    svm = SVC(C=10, kernel='rbf', probability=True)
    svm.fit(X_train, Y_train)

    Y_pred = svm.predict(X_test)
    report = classification_report(Y_test, Y_pred, target_names=label_encoder.classes_, zero_division='warn')
    jsonreport = classification_report(Y_test, Y_pred, target_names=label_encoder.classes_, output_dict=True)
    overall_accuracy = accuracy_score(Y_test, Y_pred)

    print("Classification Report:")
    print(report)
    print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")
    print(f"Features: {X_train.shape[1]}")

    with open(f'{LOGS_PATH}/ClassReports.json', 'r+') as file:
        try: data = json.load(file)
        except json.decoder.JSONDecodeError: data = {'reports':[]}

        data['reports'].append({len(data['reports'])+1:jsonreport})
        file.seek(0)
        json.dump(data, file, indent=4)
        file.truncate()
        
    return svm, overall_accuracy

def useWrapperACO(features, labels, aco: WrapperACO):
    print("Starting Ant Colony Optimization")
    solution, quality = aco.optimize()
    print("Optimization with Ant Colony Complete")
    print(f"Solution: {np.sort(solution)} with {100*quality:.2f}% accuracy")
    print(f"Ant Colony ", end=" ")
    return solution

def useWrapperPSO(features, labels, pso: WrapperPSO):
    print("Starting Particle Swarm Optimization")
    solution = pso.optimize()
    print("Optimization with Particle Swarm Complete")
    print(f"Solution: {np.sort(solution)}")
    print(f"Particle Swarm ", end=" ")
    model, accuracy = createModel(features, labels, solution)
    return model, accuracy, solution

def useSessionWrapperACO(aco: WrapperACO, fit, iterations, status, features=None, labels=None):
    match status:
        case 0: 
            solution, quality = aco.start_run(iterations)
            print(f"Solution: {np.sort(solution)} with {100*quality:.2f}% accuracy")
        case 1: 
            solution, quality = aco.continue_run(fit, iterations)
            print(f"Solution: {np.sort(solution)} with {100*quality:.2f}% accuracy")
        case 2: 
            assert features is not None and labels is not None
            solution, quality = aco.finish_run(fit, iterations)
            print(f"Solution: {np.sort(solution)} with {100*quality:.2f}% accuracy")
            model, accuracy = createModel(features, labels)
            saveModel(model, Model.AntColony, solution)