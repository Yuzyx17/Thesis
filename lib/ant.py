import numpy as np

from classifier import *

def ant_colony_optimization(features, labels, opts):

    # Parameters
    tau = 1      # pheromone value
    eta = 1      # heuristic desirability
    alpha = 1    # control pheromone
    beta = 0.1   # control heuristic
    rho = 0.2    # pheromone trail decay coefficient

    if 'N' in opts:
        N = opts['N']
    if 'T' in opts:
        max_Iter = opts['T']
    if 'tau' in opts:
        tau = opts['tau']
    if 'alpha' in opts:
        alpha = opts['alpha']
    if 'beta' in opts:
        beta = opts['beta']
    if 'rho' in opts:
        rho = opts['rho']
    if 'eta' in opts:
        eta = opts['eta']

    # Objective function
    fun = fitness_function
    # Number of dimensions
    dim = features.shape[1]
    # Initial Tau & Eta
    tau = tau * np.ones((dim, dim))
    eta = eta * np.ones((dim, dim))
    # Pre
    fitG = np.inf
    fit = np.zeros(N)

    curve = np.inf
    t = 1
    # Iterations
    while t <= max_Iter:
        # Reset ant
        X = np.zeros((N, dim))
        for i in range(N):
            # Random number of features
            num_feat = np.random.randint(1, dim + 1)
            # Ant start with random position
            X[i, 0] = np.random.randint(1, dim + 1)
            k = []
            if num_feat > 1:
                for d in range(1, num_feat):
                    # Start with previous tour
                    k.extend([X[i, d - 1]])
                    # Edge/Probability Selection (2)
                    P = (tau[k[-1], :] ** alpha) * (eta[k[-1], :] ** beta)
                    # Set selected position = 0 probability (2)
                    P[k] = 0
                    # Convert probability (2)
                    prob = P / np.sum(P)
                    # Roulette Wheel selection
                    route = random_selection(prob)
                    # Store selected position to be the next tour
                    X[i, d] = route
        # Binary
        X_bin = np.zeros((N, dim))
        for i in range(N):
            # Binary form
            ind = X[i, :].astype(int)
            ind = ind[ind != 0] - 1
            X_bin[i, ind] = 1
        # Fitness
        for i in range(N):
            # Fitness
            fit[i] = fun(features, labels, X_bin[i, :], opts)
            # Global update
            if fit[i] < fitG:
                Xgb = X[i, :]
                fitG = fit[i]
        # ---// [Pheromone update rule on tauK] //
        tauK = np.zeros((dim, dim))
        for i in range(N):
            # Update Pheromones
            tour = X[i, :]
            tour = tour[tour != 0]
            # Number of features
            len_x = len(tour)
            tour = np.append(tour, tour[0])
            for d in range(len_x):
                # Feature selected on the graph
                x = int(tour[d] - 1)
                y = int(tour[d + 1] - 1)
                # Update delta tau k on the graph (3)
                tauK[x, y] += 1 / (1 + fit[i])
        # ---// [Pheromone update rule on tauG] //
        tauG = np.zeros((dim, dim))
        tour = Xgb
        tour = tour[tour != 0]
        # Number of features
        len_g = len(tour)
        tour = np.append(tour, tour[0])
        for d in range(len_g):
            # Feature selected on the graph
            x = int(tour[d] - 1)
            y = int(tour[d + 1] - 1)
            # Update delta tau G on the graph
            tauG[x, y] = 1 / (1 + fitG)
        # ---// Evaporate pheromone // (4)
        tau = (1 - rho) * tau + tauK + tauG
        # Save
        curve[t - 1] = fitG
        print(f'\nIteration {t} Best (ACO)= {curve[t - 1]}')
        t += 1
    # Select features based on selected index
    Sf = Xgb
    Sf = Sf[Sf != 0] - 1
    # Store results
    ACO = {'sf': Sf, 'ff': features[:, Sf], 'nf': len(Sf), 'c': curve, 'f': features, 'l': labels}
    return ACO

