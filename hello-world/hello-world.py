from sklearn import tree

# DataSet
#
# Weight | Texture | Label
# 140    | smooth  | apple
# 130    | smooth  | apple
# 150    | bumpy   | orange
# 170    | bumpy   | orange

# smooth = 1
# bumpy = 0

# apple = 0
# orange = 1

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]

# Classifier: decision tree
# just an empty rules box: knows nothing about apples and oranges
classifier = tree.DecisionTreeClassifier()
# fit: finds patterns in data
classifier = classifier.fit(features, labels)

# classifies a fruit with weight 160 and bumpy (0)
# by patterns in data, the classifiers must predicts that is an orange (1)
print classifier.predict([[160, 0]])
