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
            seg_image = segment_leaf(image)
            seg_image = cv2.resize(seg_image, (FEAT_W, FEAT_H))
            images.append(seg_image)
            labels.append(class_label)
            if augment:
                for augmentation_name, augmentation_fn in AUGMENTATIONS:
                    aug_image = augmentation_fn(seg_image)
                    # seg_aug_image = segment_leaf(aug_image)
                    seg_aug_image = cv2.resize(seg_aug_image, (FEAT_W, FEAT_H))
                    images.append(seg_aug_image)
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
    scaler = StandardScaler()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=R_STATE)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_split = X_train, X_test
    Y_split = Y_train, Y_test

    def fitness_function(subset): return fitness_cv(X_train, Y_train, subset) if FOLDS > 1 else fitness(X_train, Y_train, subset)
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
        if model is not Model.BaseModel:
            fit_accuracy = fitness_cv(X_train, Y_train, subset) if FOLDS > 1 else fitness(X_train, Y_train, subset)
            print(f"Initial: {subset.shape[0]}: {fit_accuracy}")
        match model:
            case Model.BaseModel:
                classifier, accuracy = createModel(X_split, Y_split)
            case Model.AntColony:
                aco = WrapperACO(fitness_function,
                                X_train.shape[1], ants=5, iterations=5, rho=0.1, Q=.75, debug=1, accuracy=fit_accuracy, parrallel=True)
                solution = useWrapperACO(aco)
                classifier, accuracy, = createModel(X_split, Y_split, solution)
        if save:
            saveModel(classifier, scaler, model, subset)
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
                    unseen = unseen[:,solution]
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