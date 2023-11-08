WIDTH, HEIGHT = 600, 600
TESTW, TESTH = 250, 250

from sklearn.svm import SVC
MODEL_PATH = r'dataset\model\base_model.joblib'
MODEL = SVC(C=10, gamma='scale', kernel='rbf')

LTHRESHOLD = 128
DENOISE_KERNEL = (3, 3)
DENOISE_SIGMA = 0