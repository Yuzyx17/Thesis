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

print("Loading Features")
X = selected_feature_indices = np.load(f"{DATA_PATH}/features.npy")
Y = selected_feature_indices = np.load(f"{DATA_PATH}/labels.npy")
print("Features Loaded")

start = time.time()

print("Starting Training")
model = useGridSVC(X, Y, cv=5, param_grid=PARAM_GRID)
# saveSVC(model, name="base")

end = time.time()
hours, remainder = divmod(int(end-start), 3600)
minutes, seconds = divmod(remainder, 60)
elapsed = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

print(f"Training Completed\nElapsed Time: {elapsed}<00:00:00")
