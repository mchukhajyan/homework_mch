from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

data = load_breast_cancer()
X, y = data.data, data.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#KNN class implementation

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None


    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidian_distance(self, point1, point2):
        return np.sqrt(np.sum((point1-point2)**2))

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = [self.euclidian_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = np.bincount(k_nearest_labels).argmax()
            predictions.append(most_common)
        return predictions

knn_custom = KNN(k=3)


knn_custom.fit(X_train, y_train)


edictions_custom = knn_custom.predict(X_test)

sklearn_knn = KNeighborsClassifier(n_neighbors=3)
sklearn_knn.fit(X_train, y_train)
predictions_sklearn = sklearn_knn.predict(X_test)


correct_custom = sum(predictions_sklearn == y_test)
correct_sklearn = sum(predictions_sklearn == y_test)

print(f"Custom KNN classifier accuracy: {correct_custom / len(y_test):.2f}")
print(f"sklearn's KNeighborsClassifier accuracy: {correct_custom / len(y_test):.2f}")
X, y = data.data, data.target











