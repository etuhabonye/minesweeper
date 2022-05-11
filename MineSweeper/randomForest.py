from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import generateminesweeper

iris = load_iris()
target = "Safe"
features = {"Xcord", "Ycord", "Value", "Neighbors"}

df = generateminesweeper.boardTodf()
X = df[features]
y = df[target]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 100) 
 
# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)
 
# performing predictions on the test dataset
y_pred = clf.predict(X)

df2 = generateminesweeper.boardTodf()

X2 = df2[features]
y_pred2 = clf.predict(X2)
df2["Predicted"] = y_pred2

print(df2)