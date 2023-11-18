import sys

from sklearn.impute import SimpleImputer
from pre.segment import segment_leaf
from utilities.const import *

import cv2, os, joblib
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utilities.util import getFeatures

"""
Identify tool for overall comparison
"""

model_index = 1
model_name = MODELS[model_index]
model = joblib.load(f"{MODEL_PATH}/{model_name}.joblib")
class_index = 1
test = f'dataset/images/{CLASSES[class_index]}'
test = r'dataset\messenger'
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

    if model_name != MODELS[0]:
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

