import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.calibration import LabelEncoder, cross_val_predict
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC

from utilities.const import *

def custom_pso(X, y, num_particles, num_iterations, c1, c2, w, threshold):
    num_features = X.shape[1]
    lb = [0] * num_features  # Lower bound for each feature (0: not selected, 1: selected)
    ub = [1] * num_features  # Upper bound for each feature

    # Initialize particle positions and velocities
    particles = np.random.uniform(0, 1, (num_particles, num_features))
    velocities = np.zeros((num_particles, num_features)) # have zeroes as initial value of velocities
    # velocities = np.random.uniform(0, 1, (num_particles, num_features)) randomly generate initial value for velocities
    personal_best_positions = particles.copy()
    personal_best_scores = np.ones(num_particles)

    # Find the global best particle
    global_best_index = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[global_best_index]

    for iteration in range(num_iterations):
        print(f"Iteration {iteration+1}")
        for i in range(num_particles):
            # Update particle velocities
            r1, r2 = np.random.rand(2)
            velocities[i] = w * velocities[i] + c1 * r1 * (personal_best_positions[i] - particles[i]) + c2 * r2 * (global_best_position - particles[i])
            
            # Update particle positions
            particles[i] = particles[i] + velocities[i]
            
            # Clamp particle positions to the lower and upper bounds
            particles[i] = np.clip(particles[i], lb, ub)

            # Apply threshold θ to select features
            selected_features = np.where(particles[i] > threshold)[0]

            # Calculate the objective function value
            selected_features = np.where(particles[i])[0]
            if np.sum(selected_features) == 0:
                score = 1.0  # Avoid division by zero
            else:
                X_selected = X[:, selected_features]
                kfold = KFold(n_splits=FOLD, shuffle=True, random_state=42)
                classifier = SVC(kernel='rbf', C=10, probability=True)
                y_pred = cross_val_predict(classifier, X_selected, y, cv=kfold)
                accuracy = accuracy_score(y, y_pred)
                score = 1.0 - accuracy  # Minimize 1 - accuracy
            
            # Update personal best position and score
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = particles[i]
            
                # Update global best position
                if score < personal_best_scores[global_best_index]:
                    global_best_index = i
                    global_best_position = personal_best_positions[i]

        print(f"Solution at current Iteration {iteration + 1} {global_best_position}")
    return global_best_position


def TestPSO(features, labels, swarm=30, iterations=100):
    label_encoder = LabelEncoder()
    numerical_labels = label_encoder.fit_transform(labels)
    
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(features)

    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, f"{SCALER_PATH}/test-pso.pkl")

    X = scaler.transform(X)

    print("Starting PSO")
    selected_features = custom_pso(features, labels, num_particles=swarm, num_iterations=iterations, c1=1.49618, c2=1.49618, w=0.7298, threshold=0.6)
    selected_features = np.where(selected_features > 0.5)[0]
    print("Optimization with PSO Complete")

    np.save(f"{FEATURE_PATH}/test-pso.npy", selected_features)

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
# custom_pso(X, y, num_particles=30, num_iterations=100, c1=1.49618, c2=1.49618, w=0.7298, threshold=0.6)
