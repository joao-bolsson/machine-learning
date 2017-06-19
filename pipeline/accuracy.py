import random
# Class that defines a classifier
class MyClassifier():

    # Method to train the classifier
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self

    # Method to predict the data test
    def predict(self, X_test):
        predictions = []

        for row in X_test:
            label = random.choice(self.y_train)
            predictions.append(label)

        return predictions

# import a data set
from sklearn import datasets
iris = datasets.load_iris()

# f(X) = y
X = iris.data
y = iris.target

# 50% to test and 50% to train
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

classifier = MyClassifier()

# training the classifier
classifier = classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)
