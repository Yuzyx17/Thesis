import json
import math
import time, itertools
from utilities.const import *

from datetime import datetime
from pre.norm import *
from pre.segment import *
from utilities.features import *
from lib.classifier import *
from lib.WrapperACO import *

models = [Model.BaseModel, Model.ParticleSwarm, Model.AntColony]
images = []
# Generate Combinations of Features
combinations = []
for i in range(1, len(GROUPED_FEATURES) + 1):
    for subset in itertools.combinations(GROUPED_FEATURES, i):
        combination = []
        for item in subset:
            for feature_type in GROUPED_FEATURES[item]:
                combination.append(feature_type)
        combinations.append(combination)

print("Preparing Images")
labels = []
if os.path.exists(f"{DATA_PATH}/images.npy"):
    images = np.load(f"{DATA_PATH}/images.npy")
    labels = np.load(f"{DATA_PATH}/labels.npy")
else:
    for class_folder in os.listdir(DATASET_PATH):
        class_label = class_folder
        class_path = os.path.join(DATASET_PATH, class_folder)
        print(f"Class Label: {class_label}")
        if not os.path.exists(class_path):
            continue 
        # Loop through the images in the class folder
        for image_file in tqdm(os.listdir(class_path)):
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path) 
            image = segment_leaf(image)

            images.append(image)
            labels.append(class_label)

    Y = np.array(labels)
    Y = label_encoder.fit_transform(Y)

save = False
print("Starting Exhaustive Training")

for combination in combinations:
    print("Combination: ", combination)
    X = []
    for image in images:
        img_feature = []
        for feature in combination:
            img_feature.extend(FEATURES[feature](image))
        img_feature = np.array(img_feature)
        X.append(img_feature)
    X = np.array(X)
    scaler.fit(X)
    X = scaler.transform(X)

    def fitness_function(_subset): return fitness(X, Y, _subset)
    subset = np.arange(0, X.shape[1])
    fit_accuracy = 0

    if combinations.index(combination) == len(combinations) - 1:
        save = True

    for model in models:
        start = time.time()

        match model:
            case Model.BaseModel:
                classifier, accuracy = createModel(X, Y)
                fit_accuracy = accuracy
            case Model.AntColony:
                aco = WrapperACO(fitness_function,
                                X.shape[1], ants=10, iterations=15, parrallel=True, debug=1, accuracy=fit_accuracy)
                classifier, accuracy, subset = useWrapperACO(X, Y, aco)
            case Model.ParticleSwarm:
                classifier, accuracy, subset = useWrapperPSO(X, Y, swarm=10, iterations=15)
        if save:
            saveModel(classifier, model, subset)

        end = time.time()
        hours, remainder = divmod(int(end-start), 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        predictions = {
            'blb' : None,
            'hlt' : None,
            'rbl' : None,
            'sbt' : None,
        }
        diseases = ['blb', 'hlt', 'rbl', 'sbt']

        for class_folder in os.listdir(UNSEEN_PATH):
            amount = 0
            correct = 0
            curclass = diseases.index(class_folder)
            for image_file in os.listdir(f"{UNSEEN_PATH}/{class_folder}"):
                amount += 1
                image_path = os.path.join(f"{UNSEEN_PATH}/{class_folder}", image_file)
                image = cv2.imread(image_path) 
                image = segment_leaf(image)
                img_feature = []
                for feature in combination:
                    img_feature.extend(FEATURES[feature](image))
                img_feature = np.array(img_feature)
                unseen = [img_feature]
                unseen = scaler.transform(unseen)

                if model is not Model.BaseModel:
                    unseen = unseen[:,subset]
                prediction = classifier.predict(unseen)[0]
                correct += 1 if prediction == curclass else 0
  
            predictions[diseases[curclass]] = f"{(correct/amount)*100:.2f}%"


        log = {"Model": {"Name": model.name, "Date": datetime.now().strftime('%Y/%m/%d %H:%M:%S'), "Elapsed": elapsed, 'Image Size:' : f"{FEAT_W}x{FEAT_H}", "Accuracy": f"{100*accuracy:.2f}%", "Saved": "True" if save else "False",
                 'Images': X.shape[0], "Features": {'Amount': subset.shape[0], 'Feature': combination}, 'Augmentations': [aug[0] for aug in AUGMENTATIONS], "Predictions": predictions, 
                 'Additional': 'None' if model is Model.BaseModel else ({
                     'Ants': aco.ants,
                     'Iterations': aco.iterations,
                     'Rho': aco.rho,
                     'Q': aco.Q,
                     'Alpha': aco.alpha,
                     'Beta': aco.beta
                 } if model is Model.AntColony else 'None')}}
        
        with open(f'{model.name}_report.json', 'a') as logs:
            logs.write(json.dumps(log, indent=4))
            logs.write(f"\n")