import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split, GridSearchCV
import pdb

# Load digits dataset
digits = datasets.load_digits()

# Visualize first 4 training images
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# Flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Start debugging from here
# pdb.set_trace()

# Use a portion of data for hyperparameter tuning
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)

# Define parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1],
}

# Create base classifier
svc = svm.SVC()

# Perform grid search on training set
grid_search = GridSearchCV(svc, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get best estimator
best_clf = grid_search.best_estimator_
print("Best parameters found:", grid_search.best_params_)

# Predict using best model
predicted = best_clf.predict(X_test)

# Visualize predictions
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

# Print classification report
print(
    f"Classification report for classifier {best_clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

# Show confusion matrix
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()

# Rebuild classification report from confusion matrix
y_true = []
y_pred = []
cm = disp.confusion_matrix

for gt in range(len(cm)):
    for pred in range(len(cm)):
        y_true += [gt] * cm[gt][pred]
        y_pred += [pred] * cm[gt][pred]

print(
    "Classification report rebuilt from confusion matrix:\n"
    f"{metrics.classification_report(y_true, y_pred)}\n"
)

