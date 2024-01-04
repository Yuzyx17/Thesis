from const import *
from lib.Processing import *
from lib.Model import Model
from lib.Analysis import *
import pprint

X, Y = preLoadImages()
model = Model(ModelType.AntColony, X, Y)
model.load()
model.obtainConfusion(loadNamedUnseenImages()) 
model.retestModel()