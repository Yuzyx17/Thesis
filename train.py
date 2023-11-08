import sys
sys.path.append(r'lib')
sys.path.append(r'utilities')
sys.path.append(r'lib/classifier.py')
sys.path.append(r'utilities/features.py')

import cv2
import os
from pre.norm import *
from pre.segment import *
from utilities.features import *
from lib.classifier import *

dataset_dir = r'dataset\captured'
augmented_dir = r'dataset\augmented'
model_file_path = r'dataset\model'
features= []
labels= []
# Define HOG parameters
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
orient = 9
# LBP features
radius = 1
n_points = 8 * radius

sharpen_kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]], dtype=np.float32)
augmentations = [
    ('horizontal_flip', lambda img: cv2.flip(img, 1)),
    ('vertical_flip', lambda img: cv2.flip(img, 0)),
    ('rotate_90C',  lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
    ('rotate_180',  lambda img: cv2.rotate(img, cv2.ROTATE_180)),
    ('rotate_90CC', lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ('contrast_increase', lambda img: cv2.convertScaleAbs(img, alpha=1.5, beta=0)),
    ('contrast_decrease', lambda img: cv2.convertScaleAbs(img, alpha=0.5, beta=0)),
    ('brightness_increase', lambda img: cv2.convertScaleAbs(img, alpha=1, beta=50)),
    ('brightness_decrease', lambda img: cv2.convertScaleAbs(img, alpha=1, beta=-50)),
    ('blur', lambda img: cv2.GaussianBlur(img, (5, 5), 0)),  # Apply Gaussian blur
    ('sharpen', lambda img: cv2.filter2D(img, -1, sharpen_kernel)),  # Apply sharpening filter
]
# Loop through the class folders
for class_folder in os.listdir(dataset_dir):
    class_label = class_folder
    class_path = os.path.join(dataset_dir, class_folder)
    print(f"Class Label: {class_label}")
    if not os.path.exists(class_path):
        continue

    # Loop through the images in the class folder
    for image_file in os.listdir(class_path):
        image_path = os.path.join(class_path, image_file)
        image = cv2.imread(image_path) 
        image = segment_leaf(image)

        features.append(getFeatures(image))
        labels.append(class_label)

        # Apply augmentations
        # print(f'{class_label}:{image_file}')
        for augmentation_name, augmentation_fn in augmentations:
            # print(f'{class_label}:{image_file}-{augmentation_name}')

            aug_image = augmentation_fn(image)

            features.append(getFeatures(aug_image))
            labels.append(class_label)

print("Training SVC")

features = np.array(features)
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    # 'kernel': ['linear', 'poly', 'rbf'],
    'kernel' : ['rbf'],
    # 'degree': [2, 3, 4],
    'gamma': ['scale', 'auto'] + [0.01, 0.1, 1, 10],
    'coef0': [0.0, 1.0, 2.0],
    # 'shrinking': [True, False],
    'class_weight': [None, 'balanced'],
    'decision_function_shape': ['ovr', 'ovo'],
}
f = features
l = labels
saveSVC(usePSO(f, l))