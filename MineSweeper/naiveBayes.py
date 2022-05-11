from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import generateminesweeper

target = "Safe"
features = {"Xcord", "Ycord", "Value", "Neighbors"}

df = generateminesweeper.boardTodf()

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

df2 = generateminesweeper.boardTodf()

X2 = df2[features]
y_pred2 = classifer.predict(X2)
df2["Predicted"] = y_pred2

print(df2)

