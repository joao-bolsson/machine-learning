import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
# ids that will be removed from dataset to test the model accuracy later
# one flower by type
test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# Classifier: decision tree
classifier = tree.DecisionTreeClassifier()
# training the classifier with the training data
classifier = classifier.fit(train_data, train_target)

# the test target
print test_target
# the classifier prediction: output must be equals to test_target
print classifier.predict(test_data)
