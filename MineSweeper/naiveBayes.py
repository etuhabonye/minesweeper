#naive bayes algorithm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd 
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization
from sklearn.metrics import confusion_matrix
import generateminesweeper
from sklearn.naive_bayes import BernoulliNB # importing classifier
from sklearn.metrics import classification_report

target = "Safe"
features = {"Xcord", "Ycord", "Value", "Neighbors"}

df, df2 = generateminesweeper.boardTodf()

X = df[features]
Y = df[target]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
print(Y_test)
gnb = GaussianNB()

# initializaing the NB
classifer = BernoulliNB()

# training the model
classifer.fit(X_train, Y_train)

# testing the model
y_pred = classifer.predict(X)

X2 = df2[features]
y_true = df2[target]
y_pred = classifer.predict(X2)
df2["Predicted"] = y_pred

print(accuracy_score(y_true, y_pred))

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

print(classification_report(y_true, y_pred))

