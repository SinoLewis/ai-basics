## Supervised Learning

import pandas as pd
import sklearn
from sklearn import svm ,metrics, preprocessing
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


data = pd.read_csv('car.data')
columns = ["buying","maint","door","persons","lug_boot","safety","class"]
le = preprocessing.LabelEncoder()

buying, maint, door, persons, lug_boot, safety, cls = [le.fit_transform(list(data[i])) for i in columns]
X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = svm.SVC(kernel='linear')
# # SVM linear with soft margin of 2
# clf = svm.SVC(kernel='linear', C=2)
# # SVM polynomial with 3 dimensions
# clf = svm.SVC(kernel='poly', degree=3)
# # Another classifier: KNN
# clf = KNeighborsClassifier(n_neighbors=9)

clf.fit(x_train, y_train)

predictions = clf.predict(x_test)
accuracy = metrics.accuracy_score(y_test, predictions)
print(accuracy)
