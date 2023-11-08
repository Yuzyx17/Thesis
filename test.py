import sys

sys.path.append(r'lib')
sys.path.append(r'utilities')
sys.path.append(r'lib/classifier.py')

import cv2, os
from pre.norm import *
from pre.segment import *

from utilities.features import getFeatures
from utilities.util import *

model = joblib.load(MODEL_PATH)
test = r'dataset\images\sheath'
class_names = ["blb", "hlt", "rbl", "sbt"]
withPSO = True
# Loop through the images in the class folder
for image_file in os.listdir(test):
    image_path = os.path.join(test, image_file)
    image = cv2.imread(image_path) 
    image = segment_leaf(image)
    features = getFeatures(image)
    
    if withPSO:
        selected_feature_indices = np.load(r'dataset\model\selected_feature_indices.npy')

        # Apply selected features
        features = features[selected_feature_indices]
    # Make predictions using the loaded model
    predictions = model.predict([features])[0]
    print(f"Predictions: {class_names[predictions]} : {image_path}")
