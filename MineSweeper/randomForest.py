# #random forest algorithm
import numpy as np
# from collections import Counter
import generateminesweeper
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_curve, auc
import pandas as pd 
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization


# class Node:
#     '''
#     Helper class which implements a single tree node.
#     '''
#     def __init__(self, feature=None, threshold=None, data_left=None, data_right=None, gain=None, value=None):
#         self.feature = feature
#         self.threshold = threshold
#         self.data_left = data_left
#         self.data_right = data_right
#         self.gain = gain
#         self.value = value


# class DecisionTree:
#     '''
#     Class which implements a decision tree classifier algorithm.
#     '''
#     def __init__(self, min_samples_split=2, max_depth=5):
#         self.min_samples_split = min_samples_split
#         self.max_depth = max_depth
#         self.root = None

#     @staticmethod
#     def _entropy(s):
#         '''
#         Helper function, calculates entropy from an array of integer values.
#         :param s: list
#         :return: float, entropy value
#         '''
#         print("in dt entropy")
#         # Convert to integers to avoid runtime errors
#         counts = np.bincount(np.array(s, dtype=np.int64))
#         # Probabilities of each class label
#         percentages = counts / len(s)

#         # Caclulate entropy
#         entropy = 0
#         for pct in percentages:
#             if pct > 0:
#                 entropy += pct * np.log2(pct)
#         return -entropy

#     def _information_gain(self, parent, left_child, right_child):
#         '''
#         Helper function, calculates information gain from a parent and two child nodes.
#         :param parent: list, the parent node
#         :param left_child: list, left child of a parent
#         :param right_child: list, right child of a parent
#         :return: float, information gain
#         '''
#         print("in dt info gain")
#         num_left = len(left_child) / len(parent)
#         num_right = len(right_child) / len(parent)

#         # One-liner which implements the previously discussed formula
#         return self._entropy(parent) - (num_left * self._entropy(left_child) + num_right * self._entropy(right_child))

#     def _best_split(self, X, y):
#         '''
#         Helper function, calculates the best split for given features and target
#         :param X: np.array, features
#         :param y: np.array or list, target
#         :return: dict
#         '''
#         print("in dt best split")
#         best_split = {}
#         best_info_gain = -1
#         n_rows, n_cols = X.shape

#         # For every dataset feature
#         for f_idx in range(n_cols):
#             X_curr = X[:, f_idx]
#             # For every unique value of that feature
#             for threshold in np.unique(X_curr):
#                 # Construct a dataset and split it to the left and right parts
#                 # Left part includes records lower or equal to the threshold
#                 # Right part includes records higher than the threshold
#                 df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
#                 df_left = np.array([row for row in df if row[f_idx] <= threshold])
#                 df_right = np.array([row for row in df if row[f_idx] > threshold])

#                 # Do the calculation only if there's data in both subsets
#                 if len(df_left) > 0 and len(df_right) > 0:
#                     # Obtain the value of the target variable for subsets
#                     y = df[:, -1]
#                     y_left = df_left[:, -1]
#                     y_right = df_right[:, -1]

#                     # Caclulate the information gain and save the split parameters
#                     # if the current split if better then the previous best
#                     gain = self._information_gain(y, y_left, y_right)
#                     if gain > best_info_gain:
#                         best_split = {
#                             'feature_index': f_idx,
#                             'threshold': threshold,
#                             'df_left': df_left,
#                             'df_right': df_right,
#                             'gain': gain
#                         }
#                         best_info_gain = gain
#         return best_split

#     def _build(self, X, y, depth=0):
#         '''
#         Helper recursive function, used to build a decision tree from the input data.
#         :param X: np.array, features
#         :param y: np.array or list, target
#         :param depth: current depth of a tree, used as a stopping criteria
#         :return: Node
#         '''
#         print("in dt build")
#         n_rows, n_cols = X.shape

#         # Check to see if a node should be leaf node
#         if n_rows >= self.min_samples_split and depth <= self.max_depth:
#             # Get the best split
#             best = self._best_split(X, y)
#             # If the split isn't pure
#             if best['gain'] > 0:
#                 # Build a tree on the left
#                 left = self._build(
#                     X=best['df_left'][:, :-1],
#                     y=best['df_left'][:, -1],
#                     depth=depth + 1
#                 )
#                 right = self._build(
#                     X=best['df_right'][:, :-1],
#                     y=best['df_right'][:, -1],
#                     depth=depth + 1
#                 )
#                 return Node(
#                     feature=best['feature_index'],
#                     threshold=best['threshold'],
#                     data_left=left,
#                     data_right=right,
#                     gain=best['gain']
#                 )
#         # Leaf node - value is the most common target value
#         return Node(
#             value=Counter(y).most_common(1)[0][0]
#         )

#     def fit(self, X, y):
#         '''
#         Function used to train a decision tree classifier model.
#         :param X: np.array, features
#         :param y: np.array or list, target
#         :return: None
#         '''
#         print("in dt fit")
#         # Call a recursive function to build the tree
#         self.root = self._build(X, y)

#     def _predict(self, x, tree):
#         '''
#         Helper recursive function, used to predict a single instance (tree traversal).
#         :param x: single observation
#         :param tree: built tree
#         :return: float, predicted class
#         '''
#         print("in dt _predict")
#         # Leaf node
#         if tree.value != None:
#             return tree.value
#         feature_value = x[tree.feature]

#         # Go to the left
#         if feature_value <= tree.threshold:
#             return self._predict(x=x, tree=tree.data_left)

#         # Go to the right
#         if feature_value > tree.threshold:
#             return self._predict(x=x, tree=tree.data_right)

#     def predict(self, X):
#         '''
#         Function used to classify new instances.
#         :param X: np.array, features
#         :return: np.array, predicted classes
#         '''
#         print("in dt predict")
#         # Call the _predict() function for every observation
#         return [self._predict(x, self.root) for x in X]

# class RandomForest:
#     '''
#     A class that implements Random Forest algorithm from scratch.
#     '''
#     def __init__(self, num_trees=25, min_samples_split=2, max_depth=5):
#         self.num_trees = num_trees
#         self.min_samples_split = min_samples_split
#         self.max_depth = max_depth
#         # Will store individually trained decision trees
#         self.decision_trees = []

#     @staticmethod
#     def _sample(X, y):
#         '''
#         Helper function used for boostrap sampling.
#         :param X: np.array, features
#         :param y: np.array, target
#         :return: tuple (sample of features, sample of target)
#         '''
#         print("in forest sample")
#         n_rows, n_cols = X.shape
#         # Sample with replacement
#         samples = np.random.choice(a=n_rows, size=n_rows, replace=True)
#         return X[samples], y[samples]

#     def fit(self, X, y):
#         '''
#         Trains a Random Forest classifier.
#         :param X: np.array, features
#         :param y: np.array, target
#         :return: None
#         '''
#         # Reset
#         print("in forest fit")
#         if len(self.decision_trees) > 0:
#             self.decision_trees = []

#         # Build each tree of the forest
#         num_built = 0
#         while num_built < self.num_trees:
#             try:
#                 clf = DecisionTree(
#                     min_samples_split=self.min_samples_split,
#                     max_depth=self.max_depth
#                 )
#                 # Obtain data sample
#                 _X, _y = self._sample(X, y)
#                 # Train
#                 clf.fit(_X, _y)
#                 # Save the classifier
#                 self.decision_trees.append(clf)
#                 num_built += 1
#             except Exception as e:
#                 continue

#     def predict(self, X):
#         '''
#         Predicts class labels for new data instances.
#         :param X: np.array, new instances to predict
#         :return:
#         '''
#         print("in forest predict")
#         # Make predictions with every tree in the forest
#         y = []
#         for tree in self.decision_trees:
#             y.append(tree.predict(X))

#         # Reshape so we can find the most common value
#         y = np.swapaxes(a=y, axis1=0, axis2=1)

#         # Use majority voting for the final prediction
#         predictions = []
#         for preds in y:
#             counter = Counter(preds)
#             predictions.append(counter.most_common(1)[0][0])
#         return predictions



from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import generateminesweeper

iris = load_iris()
# X = iris["data"]
# y = iris["target"]
target = "Safe"
features = {"Xcord", "Ycord", "Value", "Neighbors"}

df, df2 = generateminesweeper.boardTodf()
X = df[features]
y = df[target]

X2 = df2[features]
y_true = df2[target]

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 100)

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)

# performing predictions on the test dataset
y_pred = clf.predict(X)

# df2 = generateminesweeper.boardTodf()

X2 = df2[features]
Y = df2[target]
y_pred2 = clf.predict(X2)
df2["Predicted"] = y_pred2

print(accuracy_score(y_true, y_pred2))


print(accuracy_score(y_true, y_pred))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])

# visualize confusion matrix with seaborn heatmap

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.show()

TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))

# print classification accuracy

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)

print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

# print precision score

precision = TP / float(TP + FP)


print('Precision : {0:0.4f}'.format(precision))

#print recall or sensitivity

recall = TP / float(TP + FN)

print('Recall or Sensitivity : {0:0.4f}'.format(recall))