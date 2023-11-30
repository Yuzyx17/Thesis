from const import *
from lib.Processing import *
from lib.Model import Model
from lib.Analysis import *
import pprint

image = r'dataset\finalized\validation\blb\40.jpg'
X, Y = loadUnseenImages()
model = Model(ModelType.BaseModel)
model.load()
model.obtainMetrics((X, Y))
# pprint.pprint(model.metrics)
for image_file in os.listdir(r'dataset\finalized\testing\sb'):
    image_path = os.path.join(r'dataset\finalized\testing\sb', image_file)
    print(model.predict(image_path))