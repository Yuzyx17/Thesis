import json
import time
from utilities.const import *

from datetime import datetime
from pre.norm import *
from pre.segment import *
from utilities.features import *
from lib.classifier import *
from lib.WrapperACO import *

print("Loading Features")
X = np.load(f"{DATA_PATH}/features.npy")
Y = np.load(f"{DATA_PATH}/labels.npy")
print("Features Loaded")

start = time.time()

print(f"Starting Training with Feature: {X.shape}")

# Pre process features
scaler.fit(X)
X = scaler.transform(X)
Y = label_encoder.fit_transform(Y)

# Create fitness function
def fitness_function(subset): return fitness(X, Y, subset)

save = True
model = Model.BaseModel
subset = np.arange(0, X.shape[1])
accuracy = fitness(X, Y, subset)

match model:
    case Model.BaseModel:
        classifier, accuracy = createModel(X, Y)
    case Model.AntColony:
        aco = WrapperACO(fitness_function,
                         X.shape[1], ants=2, iterations=5, debug=1, accuracy=accuracy)
        classifier, accuracy, subset = useWrapperACO(X, Y, aco)
if save:
    saveModel(classifier, model, subset)

end = time.time()
hours, remainder = divmod(int(end-start), 3600)
minutes, seconds = divmod(remainder, 60)
elapsed = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

print(f"Training Completed\nElapsed Time: {elapsed}<00:00:00")

log = {"Model": {"Name": model.name, "Date": datetime.now().strftime('%Y/%m/%d %H:%M:%S'), "Elapsed": elapsed, "Accuracy": f"{100*accuracy:.2f}%", "Saved": "True" if save else "False",
                 "Features": {'Amount': X.shape[1] if subset is None else subset.shape, 'Feature': [feature[0] for feature in FEATURES]}, 'Augmentations': [aug[0] for aug in AUGMENTATIONS], 'Image Size:' : f"{FEAT_W}x{FEAT_H}"}}



with open('logs.json', 'a') as logs:
    logs.write(json.dumps(log, indent=4))
    logs.write(f"\n")
