import sys
sys.path.append(r'lib')
sys.path.append(r'utilities')
sys.path.append(r'lib/classifier.py')
sys.path.append(r'utilities/features.py')

import cv2, os, time

from pre.norm import *
from pre.segment import *
from utilities.features import *
from lib.classifier import *
from tqdm import tqdm

features= []
labels= []

# Loop through the class folders
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

        features.append(getFeatures(image))
        labels.append(class_label)

        for augmentation_name, augmentation_fn in AUGMENTATIONS:
            aug_image = augmentation_fn(image)

            features.append(getFeatures(aug_image))
            labels.append(class_label)


features = np.array(features)
labels = np.array(labels)

print(f"Saving Features ({features.shape})")
np.save(f"{DATA_PATH}/features.npy", features)
np.save(f"{DATA_PATH}/labels.npy", labels)
print("Features Saved")