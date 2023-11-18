import json
import time
from lib.aco import WrapperAACO
from utilities.const import *

from datetime import datetime
from pre.norm import *
from pre.segment import *
from utilities.features import *
from lib.classifier import *
from lib.pso import WrapperPSO
from lib.WrapperACO import *

print("Loading Features")
X = selected_feature_indices = np.load(f"{DATA_PATH}/features.npy")
Y = selected_feature_indices = np.load(f"{DATA_PATH}/labels.npy")
print("Features Loaded")

start = time.time()

print(f"Starting Training with Feature: {X.shape}")

# Pre process features
scaler.fit(X)
X = scaler.transform(X)
Y = label_encoder.fit_transform(Y)

# Create fitness function
fitness_function = lambda subset: fitness(X, Y, subset)
subset = None

save = True
model = Model.BaseModel
accuracy = 0.0
match model:
    case Model.BaseModel: 
        classifier, accuracy = createModel(X, Y)
    case Model.AntColony: 
        aco = WrapperACO(fitness_function, X.shape[1], ants=2, iterations=5, debug=1)
        classifier, accuracy, subset = useWrapperACO(X, Y, aco)
if save:
    saveModel(classifier, model, subset)

end = time.time()
hours, remainder = divmod(int(end-start), 3600)
minutes, seconds = divmod(remainder, 60)
elapsed = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

print(f"Training Completed\nElapsed Time: {elapsed}<00:00:00")

log ={"Model": {"Name": model.name, "Date": datetime.now().strftime('%Y/%m/%d %H:%M:%S'), "Elapsed": elapsed, "Accuracy": f"{accuracy:.2f}", "Saved": "True" if save else "False"}}

with open('logs.json', 'a') as logs:
    logs.write(json.dumps(log))
    logs.write(f"\n")