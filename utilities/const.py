WIDTH, HEIGHT = 500, 500
TESTW, TESTH = 50, 50

from sklearn.svm import SVC
MODEL_PATH = r'dataset\model\models'
SCALER_PATH = r'dataset\model\scalers'
FEATURE_PATH = r'dataset\model\features'
MODEL = SVC(C=10, gamma='scale', kernel='rbf', probability=True, decision_function_shape="ovr")

MODELS = ["base", "pso", "aco", "abc"]
CLASSES = ["blb", "hlt", "rbl", "sbt"]

LTHRESHOLD = 128
DENOISE_KERNEL = (3, 3)
DENOISE_SIGMA = 0