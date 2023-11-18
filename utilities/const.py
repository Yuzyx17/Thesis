import sys

from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler

sys.path.append(r'lib')
sys.path.append(r'utilities')
sys.path.append(r'pre')

import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.svm import SVC
from utilities.features import *

label_encoder = LabelEncoder()
scaler = StandardScaler()

"""
INITIAL PAThS
"""
MODEL_PATH = r'dataset\model\models'
SCALER_PATH = r'dataset\model\scalers'
FEATURE_PATH = r'dataset\model\features'
DATA_PATH = r'dataset\model'
DATASET_PATH = r'dataset\captured'
AUG_PATH = r'dataset\augmented'
SEG_PATH = r'dataset\model'

"""
GENERAL CONSTANTS
"""

MODEL = SVC(C=10, kernel='rbf', probability=True)
MODELS = ["BaseModel", "AntColony", "ParticleSwarm"]
CLASSES = ["blb", "hlt", "rbl", "sbt"]
CORES = os.cpu_count() // 2
CORES = CORES if CORES // 2 >= os.cpu_count() else CORES + 1
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
FOLDS = 1
SHUFFLE = True

R_STATE = 42
TEST_SIZE = 0.2
"""
OBJECTIVE FUNCTION
"""
def fitness(features, labels, subset):
    selected_features = features[:, subset]

    if FOLDS > 1:
        kfold = KFold(n_splits=FOLDS, shuffle=SHUFFLE, random_state=R_STATE)
        scores = cross_val_score(MODEL, selected_features, labels, cv=kfold)
        accuracy = scores.mean()
    else:
        X_train, X_test, y_train, y_test = train_test_split(selected_features, labels, test_size=TEST_SIZE, random_state=R_STATE)
        MODEL.fit(X_train, y_train)
        y_pred = MODEL.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

    return accuracy
"""
FOR PRE-PROCESSING
"""
WIDTH, HEIGHT = 500, 500
FEAT_W, FEAT_H = 50, 50

LTHRESHOLD = 128
DENOISE_KERNEL = (3, 3)
DENOISE_SIGMA = 0

SHARPEN_KERNEL = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]], dtype=np.float32)

AUGMENTATIONS = [
    ('H_FLIP', lambda img: cv2.flip(img, 1)),
    ('V_FLIP', lambda img: cv2.flip(img, 0)),
    ('ROT90C', lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
    ('ROT180', lambda img: cv2.rotate(img, cv2.ROTATE_180)),
    ('ROT90O', lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ('CONINC', lambda img: cv2.convertScaleAbs(img, alpha=1.5, beta=0)),
    ('CONDEC', lambda img: cv2.convertScaleAbs(img, alpha=0.5, beta=0)),
    ('BR_INC', lambda img: cv2.convertScaleAbs(img, alpha=1, beta=50)),
    ('BR_DEC', lambda img: cv2.convertScaleAbs(img, alpha=1, beta=-50)),
    ('IMBLUR', lambda img: cv2.GaussianBlur(img, (5, 5), 0)),  # Apply Gaussian blur
    ('SHARPN', lambda img: cv2.filter2D(img, -1, SHARPEN_KERNEL)),  # Apply sharpening filter
]
"""
FOR FEATURE EXTRACTION
"""
# HOG (SHAPE) FEATURES
PPC = (8, 8)
CPB = (2, 2)
ORIENT = 9

# LBP (TEXTURE) FEATURES
RADIUS = 1
POINTS = 8 * RADIUS

# COLOR HISTOGRAM FEATURES
BINS = 256
CHANNELS = (0, 1, 2)
RANGES = (0, 256)

FEATURES = [
    ('H_FLIP', lambda image: getHOGFeatures(image, ORIENT, PPC, CPB)),
    ('V_FLIP', lambda image: getGLCMFeatures(image)),
    ('ROT90C', lambda image: getLBPFeatures(image)),
    ('ROT180', lambda image: getHSVFeatures(image)),
    ('ROT90O', lambda image: getLABFeatures(image)),
    ('CONINC', lambda image: getColHistFeatures(image, BINS, CHANNELS, RANGES)),
    ('CONDEC', lambda image: getCoCoFeatures(image))
]
