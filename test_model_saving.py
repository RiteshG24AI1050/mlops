import os

def test_svm_model_exists():
    assert os.path.exists("svm_model.pkl"), "SVM model not found"

def test_knn_model_exists():
    assert os.path.exists("knn_model.pkl"), "KNN model not found"
