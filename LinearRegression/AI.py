import imp
from statistics import mode
import pandas as pd

# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

df = pd.read_csv("./Data/student-mat.csv", sep=";")

data = df[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"
best = 0

X = data.drop([predict], 1)
y = data[predict]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# for _ in range(50):
#     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     model = linear_model.LinearRegression()
#     model.fit(x_train, y_train)
#     accuarcy = model.score(x_test, y_test)
#     if accuarcy > best:
#         best = accuarcy
#         print("Best accuarcy: ", best)
#         with open("student_model.pickle", "wb") as f:
#             pickle.dump(model, f)

with open("student_model.pickle", "rb") as f:
    linear = pickle.load(f)

# # Cooeficients
# print("Coefficients: ", linear.coef_)
# # interept
# print("Intercept: ", linear.intercept_, "\n")

# # Predictions
# predictions = linear.predict(x_test)
# for x in range(len(predictions)):
#     print(predictions[x], x_test.iloc[x], y_test.iloc[x])

# scatter plot of x_label = G1 and y_label = G3
style.use("ggplot")
p = "failures"
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("G3")
plt.show()