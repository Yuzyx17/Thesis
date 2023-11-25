import cv2, os, joblib

from utilities.const import *
from pre.segment import segment_leaf
from utilities.util import getFeatures

model = Model.BaseModel    
disease = Disease.sb
classifier = joblib.load(f"{MODEL_PATH}/{model.name}.joblib")
test = f'{VALIDATION_PATH}/{disease.name}'
test = r'dataset\another-google'
predictions = {
    'blb' : 0,
    'hlt' : 0,
    'rb' : 0,
    'sb' : 0,
}

amt = 0
per_image =1
print("Predicting New Images")
scaler = joblib.load(f"{SCALER_PATH}/{model.name}.pkl")
# Loop through the images in the class folder
for image_file in os.listdir(test):
    amt += 1
    image_path = os.path.join(test, image_file)
    image = cv2.imread(image_path) 
    image = segment_leaf(image)
    image = cv2.resize(image, (FEAT_W, FEAT_H))
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
            predictions['rb'] += 1
            prediction = 'rb'
         case 3: 
            predictions['sb'] += 1
            prediction = 'sb'

    if per_image:
        print(f"Image: {image_file} | Prediction: {prediction}")

print(f"Predictions: on \"{test}\" path with Model: {model.name} [Total: {amt}] Features: {X.shape}")
for prediction in predictions:
    curclass = predictions[prediction]
    print(f"{prediction}: {curclass} [{int(curclass/amt * 100)}%]")