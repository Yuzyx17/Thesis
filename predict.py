import sys

from sklearn.impute import SimpleImputer

sys.path.append(r'lib')
sys.path.append(r'utilities')
sys.path.append(r'lib/classifier.py')

import cv2, os
from pre.norm import *
from pre.segment import *
from sklearn.preprocessing import StandardScaler
from utilities.features import getFeatures
from utilities.util import *
from utilities.const import *
from tqdm import tqdm


model_index = 4
model_name = MODELS[model_index]
model = joblib.load(f"{MODEL_PATH}/{model_name}.joblib")
class_index = 3
test = f'dataset/images/{CLASSES[class_index]}'
predictions = {
    'blb' : 0,
    'hlt' : 0,
    'rbl' : 0,
    'sbt' : 0,
}
amt = 0
per_image = 0
print("Predicting New Images")
# Loop through the images in the class folder
for image_file in os.listdir(test):
    amt += 1
    image_path = os.path.join(test, image_file)
    image = cv2.imread(image_path) 
    image = segment_leaf(image)
    X = [getFeatures(image)]
    
    scaler = joblib.load(f"{SCALER_PATH}/{model_name}.pkl")
    X = scaler.transform(X)

    if model_name != "base":
        selected_features = np.load(f"{FEATURE_PATH}/{model_name}.npy")
        X = X[:,selected_features]

    prediction = model.predict(X)[0]
    match prediction:
         case 0: predictions['blb'] += 1
         case 1: predictions['hlt'] += 1
         case 2: predictions['rbl'] += 1
         case 3: predictions['sbt'] += 1

    if per_image:
        print(f"Image: {image_file} | Prediction: {CLASSES[prediction]}")

print(f"Predictions: on \"{test}\" path with Model: {MODELS[model_index]} [Total: {amt}] Features: {X.shape}")
for prediction in predictions:
    curclass = predictions[prediction]
    print(f"{prediction}: {curclass} [{int(curclass/amt * 100)}%]")

