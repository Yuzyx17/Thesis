import json
import time, joblib
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
print(f"Features Loaded ({X.shape})")


# Pre process features
scaler.fit(X)
X = scaler.transform(X)
Y = label_encoder.fit_transform(Y)
joblib.dump(label_encoder, r'dataset\model\encoder.joblib')

# Create fitness function
def fitness_function(subset): return fitness(X, Y, subset)

save = True
model = Model.ParticleSwarm
subset = np.arange(0, X.shape[1])
accuracy = 0
if model is not Model.BaseModel:
    accuracy = fitness(X, Y, subset)

start = time.time()
match model:
    case Model.BaseModel:
        classifier, accuracy = createModel(X, Y)
    case Model.AntColony:
        aco = WrapperACO(fitness_function,
                         X.shape[1], ants=5, iterations=10, debug=1, accuracy=accuracy, parrallel=True)
        classifier, accuracy, subset = useWrapperACO(X, Y, aco)
    case Model.ParticleSwarm:
        classifier, accuracy, subset = useWrapperPSO(X, Y, 2, 5)
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
