import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, 
    recall_score,
    precision_score,
    confusion_matrix, 
    classification_report,
    mean_squared_error
)
from sklearn.model_selection import cross_val_score


def rmse(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return rmse


def rmse_scores(model, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train,
                         scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)

    print('Training data RMSE mean:\n\t', np.mean(rmse_scores), '\n')
    print('Training data RMSE standard Deviation:\n\t', np.std(rmse_scores), '\n')


def adjust_class(pred, t):
    return [1 if y >= t else 0 for y in pred]


def model_report(y_test, y_pred):
    print('Accuracy:\n\t', accuracy_score(y_test, y_pred), '\n')
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred), '\n')
    print('Classification report:\n', classification_report(y_test, y_pred), '\n')

    
    print('RMSE:\n\t', rmse(y_test, y_pred), '\n')


def model_score(y_test, y_pred, model, label, t=0.5):
    return {
        'model': model,
        'label': label,
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'rmse': rmse(y_test, y_pred),
        'threshold': t
        }


def plot_roc_curve(fpr, tpr, label):
    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)

    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')

    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    plt.show()


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", linewidth=2, label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", linewidth=2, label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')