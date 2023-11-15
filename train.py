import sys

from lib.pso import TestPSO

sys.path.append(r'lib')
sys.path.append(r'utilities')
sys.path.append(r'lib/classifier.py')
sys.path.append(r'utilities/features.py')

import cv2, os, time

from pre.norm import *
from pre.segment import *
from utilities.features import *
from lib.classifier import *
from lib.aco import TestACO
from tqdm import tqdm

print("Loading Features")
X = selected_feature_indices = np.load(f"{DATA_PATH}/features.npy")
Y = selected_feature_indices = np.load(f"{DATA_PATH}/labels.npy")
print("Features Loaded")

start = time.time()

print(f"Starting Training with Feature: {X.shape}")
save = True
model_index = 3
match model_index:
    case 0: model = useBase(X, Y)
    case 1: model = usePSO(X, Y, swarm=2, iterations=2)
    case 2: model = useACO(X, Y)
    case 3: model = TestACO(X, Y, parallel=True, iterations=50)
    case 4: model = TestPSO(X, Y, iterations=50) 
if save:
    saveSVC(model, name=MODELS[model_index])

end = time.time()
hours, remainder = divmod(int(end-start), 3600)
minutes, seconds = divmod(remainder, 60)
elapsed = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

print(f"Training Completed\nElapsed Time: {elapsed}<00:00:00")
