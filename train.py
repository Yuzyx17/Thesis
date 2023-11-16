import json
import sys

sys.path.append(r'lib')
sys.path.append(r'utilities')
sys.path.append(r'lib/classifier.py')
sys.path.append(r'utilities/features.py')

import time

from datetime import datetime
from pre.norm import *
from pre.segment import *

from utilities.features import *

from lib.classifier import *
from lib.pso import WrapperPSO
from lib.aco import *

print("Loading Features")
X = selected_feature_indices = np.load(f"{DATA_PATH}/features.npy")
Y = selected_feature_indices = np.load(f"{DATA_PATH}/labels.npy")
print("Features Loaded")

start = time.time()

print(f"Starting Training with Feature: {X.shape}")
save = 1
model_index = 0
accuracy = 0
match model_index:
    case 0: model, accuracy = BaseModel(X, Y)
    case 1: model, accuracy = WrapperACO(X, Y, parallel=True)
    case 2: model, accuracy = WrapperPSO(X, Y, swarm=20, iterations=50) 
if save:
    saveSVC(model, name=MODELS[model_index])

end = time.time()
hours, remainder = divmod(int(end-start), 3600)
minutes, seconds = divmod(remainder, 60)
elapsed = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

print(f"Training Completed\nElapsed Time: {elapsed}<00:00:00")

log ={"Model": {"Name": MODELS[model_index], "Date": datetime.now().strftime('%Y/%m/%d %H:%M:%S'), "Elapsed": elapsed, "Accuracy": f"{accuracy:.2f}", "Saved": "True" if save else "False"}}

with open('logs.json', 'w') as logs:
    json.dump(log, logs)