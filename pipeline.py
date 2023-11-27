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
from utilities.util import saveModel

models = [Model.BaseModel]
images = []
mark = datetime.now().strftime('%Y%m%d-%H%M%S')
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
    print("Images Loaded")
    images = np.load(f"{DATA_PATH}/images.npy")
    labels = np.load(f"{DATA_PATH}/labels.npy")
else:
    path = AUGMENTED_PATH
    augment = False
    print("Processing Images")
    for class_folder in os.listdir(path):
        class_label = class_folder
        class_path = os.path.join(path, class_folder)
        print(f"Class Label: {class_label}")
        if not os.path.exists(class_path):
            continue 
        # Loop through the images in the class folder
        for image_file in tqdm(os.listdir(class_path)):
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path) 
            # if path is not SEGMENTED_PATH and path is not AUGMENTED_PATH:
            # image = segment_leaf(image)
            image = cv2.resize(image, (FEAT_W, FEAT_H))
            images.append(image)
            labels.append(class_label)
            if augment:
                for augmentation_name, augmentation_fn in AUGMENTATIONS:
                    aug_image = augmentation_fn(image)
                    # if path is not SEGMENTED_PATH or path is not AUGMENTED_PATH:
                    #     aug_image = segment_leaf(aug_image)
                    # seg_aug_image = cv2.resize(aug_image, (FEAT_W, FEAT_H))
                    images.append(aug_image)
                    labels.append(class_label)

    Y = np.array(labels)
    Y = label_encoder.fit_transform(Y)
print(f"Images: {Y.shape[0]}")
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
        try:
            with open(f'{LOGS_PATH}/{model.name}/{model.name}-{mark}.json', 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            with open(f'{LOGS_PATH}/{model.name}/{model.name}-{mark}.json', 'w') as file:
                data = {'tests': []}
                json.dump(data, file, indent=4)

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
            exec(open("predict.py").read())

        end = time.time()
        hours, remainder = divmod(int(end-start), 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        diseases = ['blb', 'hlt', 'rb', 'sb']
        testing = {
            'blb' : None,
            'hlt' : None,
            'rb' : None,
            'sb' : None,
        }
        validation = {
            'blb' : None,
            'hlt' : None,
            'rb' : None,
            'sb' : None,
        }

        for class_folder in os.listdir(TESTING_PATH):
            amount = 0
            correct = 0
            curclass = diseases.index(class_folder)
            for image_file in os.listdir(f"{TESTING_PATH}/{class_folder}"):
                amount += 1
                image_path = os.path.join(f"{TESTING_PATH}/{class_folder}", image_file)
                test_image = cv2.imread(image_path) 
                test_image = segment_leaf(test_image)
                test_image = cv2.resize(test_image, (FEAT_W, FEAT_H))
                img_feature = []
                for feature in combination:
                    img_feature.extend(FEATURES[feature](test_image))
                img_feature = np.array(img_feature)
                unseen = [img_feature]
                unseen = scaler.transform(unseen)

                if model is not Model.BaseModel:
                    unseen = unseen[:,subset]
                prediction = classifier.predict(unseen)[0]
                correct += 1 if prediction == curclass else 0
  
            testing[diseases[curclass]] = f"{(correct/amount)*100:.2f}%"


        for class_folder in os.listdir(VALIDATION_PATH):
            amount = 0
            correct = 0
            curclass = diseases.index(class_folder)
            for image_file in os.listdir(f"{VALIDATION_PATH}/{class_folder}"):
                amount += 1
                image_path = os.path.join(f"{VALIDATION_PATH}/{class_folder}", image_file)
                test_image = cv2.imread(image_path) 
                test_image = segment_leaf(test_image)
                test_image = cv2.resize(test_image, (FEAT_W, FEAT_H))
                img_feature = []
                for feature in combination:
                    img_feature.extend(FEATURES[feature](test_image))
                img_feature = np.array(img_feature)
                unseen = [img_feature]
                unseen = scaler.transform(unseen)

                if model is not Model.BaseModel:
                    unseen = unseen[:,subset]
                prediction = classifier.predict(unseen)[0]
                correct += 1 if prediction == curclass else 0
  
            validation[diseases[curclass]] = f"{(correct/amount)*100:.2f}%"

        log = {f"Test-{len(data['tests'])+1}": 
               {"Name": model.name, 
                "Date": datetime.now().strftime('%Y/%m/%d %H:%M:%S'), 
                "Elapsed": elapsed, 
                'Image Size:' : f"{FEAT_W}x{FEAT_H}", 
                "Accuracy": f"{100*accuracy:.2f}%", 
                "Saved": "True" if save else "False",
                'Images': X.shape[0], 
                "Features": 
                    {'Amount': subset.shape[0], 
                     'Feature': combination},  
                "Testing (LB)": testing, 
                "Validation (OL)": validation, 
                'Additional': 'None' if model is Model.BaseModel else ({
                    'Ants': aco.ants,
                    'Iterations': aco.iterations,
                    'Rho': aco.rho,
                    'Q': aco.Q,
                    'Alpha': aco.alpha,
                    'Beta': aco.beta
                } if model is Model.AntColony else 'None')}}
        
        with open(f'{LOGS_PATH}/{model.name}/{model.name}-{mark}.json', 'w+') as file:
            data['tests'].append(log)
            json.dump(data, file, indent=4)