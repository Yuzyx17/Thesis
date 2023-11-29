from const import *
from lib.Processing import *
from lib.Model import Model


image = r'dataset\finalized\validation\blb\40.jpg'
X, Y = preLoadImages() # Applies pre-processing and feature extraction
model = Model(ModelType.AntColony)
model.load()
prediction = model.predict(image)
print(prediction)

