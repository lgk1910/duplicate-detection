import numpy as np
import pickle
import glob
import random
from sklearn.neighbors import KNeighborsClassifier

def log_transform(X):
    return np.log(X + X.min() + 1)

class KMeans():
    def __init__(self):
        with open('centroids.pickle', 'rb') as f:
            self.centroids = pickle.load(f)
        with open('labels.pickle', 'rb') as f:
            labels = pickle.load(f)
        
        self.knn = KNeighborsClassifier(n_neighbors=1)
        self.knn.fit(self.centroids, labels)

    def predict(self, X):
        return self.knn.predict(X)