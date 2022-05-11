from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_curve, auc
import pandas as pd 
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization

import generateminesweeper

target = "Safe"
features = {"Xcord", "Ycord", "Value", "Neighbors"}

df, df2 = generateminesweeper.boardTodf()

X = df[features]
Y = df[target]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
print(Y_test)
gnb = GaussianNB()
# print(gnb.fit(X_train, Y_train).predict(X_test))
#Y_gnb_score = gnb.predict_proba(X_test)


# importing classifier
from sklearn.naive_bayes import BernoulliNB

# initializaing the NB
classifer = BernoulliNB()

# training the model
classifer.fit(X_train, Y_train)

# testing the model
y_pred = classifer.predict(X)

# df2 = generateminesweeper.boardTodf()

X2 = df2[features]
y_true = df2[target]
y_pred = classifer.predict(X2)
df2["Predicted"] = y_pred

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

