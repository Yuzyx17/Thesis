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


model_index = 0
model_name = MODELS[model_index]
model = joblib.load(f"{MODEL_PATH}/{model_name}.joblib")
test = r'dataset\images\bacterial'

# Loop through the images in the class folder
for image_file in os.listdir(test):
    image_path = os.path.join(test, image_file)
    image = cv2.imread(image_path) 
    image = segment_leaf(image)
    X = [getFeatures(image)]
    
    scaler = joblib.load(f"{SCALER_PATH}/{model_name}.pkl")
    X = scaler.transform(X)

    if model_name != "base":
        selected_feature_indices = np.load(f"{FEATURE_PATH}/{model_name}.npy")
        X = X[selected_feature_indices]

    predictions = model.predict(X)[0]
    print(f"Prediction: {CLASSES[predictions]} : {image_path}")
