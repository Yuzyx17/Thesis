WIDTH, HEIGHT = 500, 500
FEAT_W, FEAT_H = 50, 50

import os
import cv2
import numpy as np
from sklearn.svm import SVC
MODEL_PATH = r'dataset\model\models'
SCALER_PATH = r'dataset\model\scalers'
FEATURE_PATH = r'dataset\model\features'
DATA_PATH = r'dataset\model'
DATASET_PATH = r'dataset\captured'
AUG_PATH = r'dataset\augmented'
SEG_PATH = r'dataset\model'

MODEL = SVC(C=10, kernel='rbf', probability=True)

MODELS = ["base", "pso", "aco", "AntColony", "test-pso"]
CLASSES = ["blb", "hlt", "rbl", "sbt"]

PARAM_GRID = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf'],
    'gamma': ['scale', 'auto'] + [0.1, 1],
    # 'coef0': [0.0, 2.0],
    'class_weight': [None, 'balanced'],
    'decision_function_shape': ['ovr', 'ovo'],
    # 'shrinking': [True, False],
    'probability': [True, False],  # Add this line to control the randomness of the underlying implementation
    # 'random_state': [None, 0, 42]  # Add this line to control the seed of the random number generator
}
CORES = os.cpu_count() // 2
CORES = CORES if CORES // 2 >= os.cpu_count() else CORES + 1
FOLD = 3

LTHRESHOLD = 128
DENOISE_KERNEL = (3, 3)
DENOISE_SIGMA = 0

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
AUGMENTATIONS = [
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