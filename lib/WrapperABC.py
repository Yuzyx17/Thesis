import os
import joblib
import numpy as np
import numpy.typing as npt

from typing import List
from joblib import Parallel, delayed
from const import *

class WrapperABC():
    ...