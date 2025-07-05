import joblib
from sklearn import datasets, svm, neighbors
from sklearn.model_selection import train_test_split

# Load digits dataset
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

# Train SVM
svm_clf = svm.SVC(gamma=0.001, C=10.0)
svm_clf.fit(X_train, y_train)
joblib.dump(svm_clf, "svm_model.pkl")

# Train KNN
knn_clf = neighbors.KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
joblib.dump(knn_clf, "knn_model.pkl")

print("Models trained and saved successfully.")
