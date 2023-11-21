import cv2, os, joblib

from pre.segment import segment_leaf
from utilities.util import getFeatures
from utilities.const import *

"""
Identify tool for overall comparison
"""

model = Model.BaseModel
disease = Disease.rbl
classifier = joblib.load(f"{MODEL_PATH}/{model.name}.joblib")
test = f'{UNSEEN_PATH}/{disease.name}'
# test = r'dataset\messenger\rbl'
predictions = {
    'blb' : 0,
    'hlt' : 0,
    'rbl' : 0,
    'sbt' : 0,
}

amt = 0
per_image = 0
print("Predicting New Images")
scaler = joblib.load(f"{SCALER_PATH}/{model.name}.pkl")
# Loop through the images in the class folder
for image_file in os.listdir(test):
    amt += 1
    image_path = os.path.join(test, image_file)
    image = cv2.imread(image_path) 
    image = segment_leaf(image)
    X = [getFeatures(image)]
    
    X = scaler.transform(X)

    if model is not Model.BaseModel:
        selected_features = np.load(f"{FEATURE_PATH}/{model.name}.npy")
        X = X[:,selected_features]

    prediction = classifier.predict(X)[0]
    match prediction:
         case 0: 
            predictions['blb'] += 1
            prediction = 'blb'
         case 1: 
            predictions['hlt'] += 1
            prediction = 'hlt'
         case 2: 
            predictions['rbl'] += 1
            prediction = 'rbl'
         case 3: 
            predictions['sbt'] += 1
            prediction = 'sbt'

    if per_image:
        print(f"Image: {image_file} | Prediction: {prediction}")

print(f"Predictions: on \"{test}\" path with Model: {model.name} [Total: {amt}] Features: {X.shape}")
for prediction in predictions:
    curclass = predictions[prediction]
    print(f"{prediction}: {curclass} [{int(curclass/amt * 100)}%]")

