from collections import Counter
from const import *
from lib.Processing import *
from lib.Model import Model
from lib.Analysis import *

image = r'dataset\finalized\validation\blb\40.jpg'
X, Y = preLoadImages() # Applies pre-processing and feature extraction
model = Model(ModelType.BaseModel, X, Y)
model.load()
model.obtainMetrics()
print(model.report, '\n',model.confusion, '\n',model.metrics)