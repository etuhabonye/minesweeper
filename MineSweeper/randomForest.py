# #random forest algorithm
import numpy as np
import generateminesweeper
from sklearn.metrics import accuracy_score
import pandas as pd 
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import generateminesweeper

target = "Safe"
features = {"Xcord", "Ycord", "Value", "Neighbors"}

df, df2 = generateminesweeper.boardTodf()
X = df[features]
y = df[target]

X2 = df2[features]
y_true = df2[target]

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