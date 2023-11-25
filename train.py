import json
import time, joblib
from utilities.const import *

from datetime import datetime
from pre.norm import *
from pre.segment import *
from utilities.features import *
from lib.classifier import *
from lib.WrapperACO import *
from utilities.util import getFeatures, saveModel

exec(open("extract_feature.py").read())

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
model = Model.BaseModel
subset = np.arange(0, X.shape[1])
accuracy = 0
if model is not Model.BaseModel:
    accuracy = fitness(X, Y, subset)


try:
    with open(f'{LOGS_PATH}/logs.json', 'r') as file:
        data = json.load(file)
except FileNotFoundError:
    with open(f'{LOGS_PATH}/logs.json', 'w') as file:
        data = {'logs': []}
        json.dump(data, file, indent=4)

start = time.time()
print(f"Training {model.name}")
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
predictions = {
        'blb' : None,
        'hlt' : None,
        'rb' : None,
        'sb' : None,
    }

for class_folder in os.listdir(VALIDATION_PATH):
    amount = 0
    correct = 0
    curclass = DISEASES.index(class_folder)
    for image_file in os.listdir(f"{VALIDATION_PATH}/{class_folder}"):
        amount += 1
        image_path = os.path.join(f"{VALIDATION_PATH}/{class_folder}", image_file)
        image = cv2.imread(image_path) 
        image = segment_leaf(image)
        image = cv2.resize(image, (FEAT_W, FEAT_H))
        unseen = [getFeatures(image)]
        unseen = scaler.transform(unseen)

        if model is not Model.BaseModel:
            unseen = unseen[:,subset]
        prediction = classifier.predict(unseen)[0]
        correct += 1 if prediction == curclass else 0
    predictions[DISEASES[curclass]] = f"{(correct/amount)*100:.2f}%"

log = {f"Model-{len(data['logs'])+1}": {"Name": model.name, "Date": datetime.now().strftime('%Y/%m/%d %H:%M:%S'), "Elapsed": elapsed, 'Image Size:' : f"{FEAT_W}x{FEAT_H}", "Accuracy": f"{100*accuracy:.2f}%", "Saved": "True" if save else "False",
                'Images': X.shape[0], "Features": subset.shape[0], 
            #  'Augmentations': [aug[0] for aug in AUGMENTATIONS], 
                "Predictions": predictions, 
                'Additional': 'None' if model is Model.BaseModel else ({
                    'Ants': aco.ants,
                    'Iterations': aco.iterations,
                    'Rho': aco.rho,
                    'Q': aco.Q,
                    'Alpha': aco.alpha,
                    'Beta': aco.beta
                } if model is Model.AntColony else 'None')}}
with open(f'{LOGS_PATH}/logs.json', 'w+') as file:
    data['logs'].append(log)
    json.dump(data, file, indent=4)
# with open(f'{LOGS_PATH}/logs.json', 'a') as logs:
#     logs.write(json.dumps(log, indent=4))
#     logs.write(f"\n")
