import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



# Example usage with random data and multiclass labels
np.random.seed(42)
X = np.random.rand(100, 10)
y = np.random.randint(3, size=100)  # 3 classes for multiclass

