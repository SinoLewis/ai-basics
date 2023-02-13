# UNSUPERVISED LEARNING

import pandas as pd
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, linear_model
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('car.data')
columns = ["buying","maint","door","persons","lug_boot","safety","class"]
le = preprocessing.LabelEncoder()

buying, maint, door, persons, lug_boot, safety, cls = [le.fit_transform(list(data[i])) for i in columns]
X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
predictions = model.predict(x_test)
names = ['unacc', 'acc', 'good', 'vgood']

for x in range(len(predictions)):
    print(f"Prediction: {names[predictions[x]]}, Actual x: {x_test[x]}, Actual y: {names[y_test[x]]}")
    n = model.kneighbors([x_test[x]], return_distance=True)
    pprint(f"Nearest Neighbors: {n[1][0]}")
