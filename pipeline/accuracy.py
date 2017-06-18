# import a data set
from sklearn import datasets
iris = datasets.load_iris()

# f(X) = y
X = iris.data
y = iris.target

# 50% to test and 50% to train
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

from sklearn import tree
classifier = tree.DecisionTreeClassifier()

# training the classifier
classifier = classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)
