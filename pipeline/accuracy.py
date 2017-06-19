from scipy.spatial import distance
# Class that defines a classifier
class MyClassifier():

    # distance between two data
    def euc(self, a, b):
        return distance.euclidean(a, b)

    # Method to train the classifier
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self

    # Method to predict the data test
    def predict(self, X_test):
        predictions = []

        for row in X_test:
            label = self.closest(row)
            predictions.append(label)

        return predictions

    def closest(self, row):
        best_distance = self.euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = self.euc(row, self.X_train[i])
            if dist < best_distance:
                best_distance = dist
                best_index = i

        return self.y_train[best_index]

# import a data set
from sklearn import datasets
iris = datasets.load_iris()

# f(X) = y
X = iris.data
y = iris.target

# 50% to test and 50% to train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

classifier = MyClassifier()

# training the classifier
classifier = classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)
