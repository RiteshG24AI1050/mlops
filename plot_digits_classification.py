# plot_digits_classification.py

from utils import (
    load_digits_data, plot_training_images, flatten_images, split_data,
    train_classifier, predict, plot_predictions, print_classification_report,
    plot_confusion_matrix, rebuild_classification_report_from_cm,
    show_plots, tune_hyperparameters  # Added tune_hyperparameters
)

def main():
    digits = load_digits_data()
    plot_training_images(digits)
    data = flatten_images(digits)
    X_train, X_test, y_train, y_test = split_data(data, digits.target)
    # Use hyperparameter tuning instead of fixed training
    clf = tune_hyperparameters(X_train, y_train)

    predicted = predict(clf, X_test)
    plot_predictions(X_test, predicted)
    print_classification_report(y_test, predicted, clf)
    disp = plot_confusion_matrix(y_test, predicted)
    rebuild_classification_report_from_cm(disp)
    show_plots()

if __name__ == "__main__":
    main()
