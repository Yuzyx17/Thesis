import json
import time, joblib
from utilities.const import *

from datetime import datetime
from pre.norm import *
from pre.segment import *
from utilities.features import *
from lib.classifier import *
from lib.WrapperACO import *
from lib.WrapperPSO import WrapperPSO
from utilities.util import getFeatures, saveModel

exec(open("extract_feature.py").read())
scaler = StandardScaler()

print("Loading Features")
X = np.load(f"{DATA_PATH}/features.npy")
Y = np.load(f"{DATA_PATH}/labels.npy")
Y = label_encoder.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_split = X_train, X_test
Y_split = Y_train, Y_test
print(f"Features Loaded {X.shape}")

# Create fitness function
def fitness_function(subset): return fitness_cv(X_train, Y_train, subset) if FOLDS > 1 else fitness(X_train, Y_train, subset)
def fitness_pso_function(subset): return fitness_pso(X, Y, subset)

save = True
model = Model.BaseModel
subset = np.arange(0, X_train.shape[1])
fit_accuracy = 0
if model is not Model.BaseModel:
    fit_accuracy = fitness_cv(X_train, Y_train, subset) if FOLDS > 1 else fitness(X_train, Y_train, subset)
    print(f"Initial: {subset.shape[0]}: {fit_accuracy}")

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
        classifier, accuracy = createModel(X_split, Y_split)
    case Model.AntColony:
        aco = WrapperACO(fitness_function,
                         X_train.shape[1], ants=5, iterations=5, rho=0.1, Q=.75, debug=1, accuracy=fit_accuracy, parrallel=True)
        solution = useWrapperACO(aco)
        classifier, accuracy, = createModel(X_split, Y_split, solution)
    case Model.ParticleSwarm:
        pso = WrapperPSO(fitness_pso_function, X.shape[1], particles=2, iterations=5)
        classifier, accuracy, subset = useWrapperPSO(X, Y, pso)
if save:
    saveModel(classifier, scaler, model, subset)

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

for class_folder in os.listdir(TESTING_PATH):
    amount = 0
    correct = 0
    curclass = DISEASES.index(class_folder)
    for image_file in os.listdir(f"{TESTING_PATH}/{class_folder}"):
        amount += 1
        image_path = os.path.join(f"{TESTING_PATH}/{class_folder}", image_file)
        image = cv2.imread(image_path) 
        image = segment_leaf(image)
        image = cv2.resize(image, (FEAT_W, FEAT_H))
        unseen = [getFeatures(image)]
        unseen = scaler.transform(unseen)

        if model is not Model.BaseModel:
            unseen = unseen[:,solution]
        prediction = classifier.predict(unseen)[0]
        correct += 1 if prediction == curclass else 0
    predictions[DISEASES[curclass]] = f"{(correct/amount)*100:.2f}%"

log = {f"Model-{len(data['logs'])+1}": 
       {"Name": model.name, 
        "Date": datetime.now().strftime('%Y/%m/%d %H:%M:%S'), 
        "Elapsed": elapsed, 
        "Image Size:" : f"{FEAT_W}x{FEAT_H}", 
        "Accuracy": f"{100*accuracy:.2f}%", 
        "Saved": "True" if save else "False",
        "Images": X_train.shape[0] + X_test.shape[0], "Features": subset.shape[0], 
        "Predictions (Test Set)": predictions, 
        "Additional": 'None' if model is Model.BaseModel else ({
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
