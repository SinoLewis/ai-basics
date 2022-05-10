import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()
print(cancer.feature_names)
print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

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