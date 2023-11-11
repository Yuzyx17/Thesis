WIDTH, HEIGHT = 500, 500
FEAT_W, FEAT_H = 50, 50

from sklearn.svm import SVC
MODEL_PATH = r'dataset\model\models'
SCALER_PATH = r'dataset\model\scalers'
FEATURE_PATH = r'dataset\model\features'
DATA_PATH = r'dataset\model'
DATASET_PATH = r'dataset\captured'
AUG_PATH = r'dataset\augmented'
SEG_PATH = r'dataset\model'

MODEL = SVC(C=0.01, gamma='scale', kernel='rbf', decision_function_shape="ovo")

MODELS = ["base", "pso", "aco", "abc"]
CLASSES = ["blb", "hlt", "rbl", "sbt"]

PARAM_GRID = {
    'C': [0.01, 0.1, 1, 10],
    'kernel': ['rbf'],
    'gamma': ['scale', 'auto'] + [0.1, 5],
    'coef0': [0.0, 2.0],
    'class_weight': [None, 'balanced'],
    'decision_function_shape': ['ovr', 'ovo'],
    # 'shrinking': [True, False],
    'probability': [True, False],  # Add this line to control the randomness of the underlying implementation
    # 'random_state': [None, 0, 42]  # Add this line to control the seed of the random number generator
}

LTHRESHOLD = 128
DENOISE_KERNEL = (3, 3)
DENOISE_SIGMA = 0