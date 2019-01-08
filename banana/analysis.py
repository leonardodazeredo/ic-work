import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils.misc import myround
from mlxtend.plotting import plot_decision_regions, plot_confusion_matrix


def print_results_svm(data, grid, y_true, y_pred):
    print("#################################################")
    print("{}-fold".format(grid.n_splits_))
    print("#################################################")
    print("Mean accuracy of CV:")
    print()
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    i = 0
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        i += 1
        print("%d > E_val: %0.3f (+/-%0.03f) for %r" % (i, 1 - mean, std ** 2, params))
    print()
    print("Best: %d > E_val: %0.3f for %r" % (grid.best_index_ + 1, 1 - means[grid.best_index_], grid.best_params_))
    print()
    print("-----------------------------------------------------------")
    print("Details of model with best parameters set on test set:")
    print()
    n_sv = grid.best_estimator_.support_vectors_.shape[0]
    max_error = myround(n_sv / (data['X_train'].shape[0] - 1))
    print("Number of SVs:", n_sv)
    print("SV / N (Upper bound on E_out):", max_error, "(Accuracy at least {})".format(myround(1 - max_error)))
    print()
    acc = myround(accuracy_score(y_true, y_pred))
    print("Accuracy: {} (E_out: {})".format(acc, myround(1 - acc)))
    print()
    print(classification_report(y_true, y_pred))


def plot_svm(data, grid, y_true, y_pred):
    X_plot = data['X_train'].values
    y_plot = data['y_train'].values.astype(np.integer)
    plot_decision_regions(X_plot, y_plot, clf=grid.best_estimator_, legend=2, scatter_kwargs=dict(s=20), markers='+o')

    sv = pd.DataFrame(grid.best_estimator_.support_vectors_, columns=['At1', 'At2'])
    plt.scatter(sv.At1, sv.At2, s=80, facecolors='none', edgecolors='black')

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plot_confusion_matrix(conf_mat=cm, show_absolute=True, show_normed=True, colorbar=True)


def print_results_knn(data, grid, y_true, y_pred):
    print("#################################################")
    print("{}-fold".format(grid.n_splits_))
    print("#################################################")
    print("Mean accuracy of CV:")
    print()
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    i = 0
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        i += 1
        print("%d > E_val: %0.3f (+/-%0.03f) for %r" % (i, 1 - mean, std ** 2, params))
    print()
    print("Best: %d > E_val: %0.3f for %r" % (grid.best_index_ + 1, 1 - means[grid.best_index_], grid.best_params_))
    print()
    print("-----------------------------------------------------------")
    print("Details of model with best parameters set on test set:")
    print()
    acc = myround(accuracy_score(y_true, y_pred))
    print("Accuracy: {} (E_out: {})".format(acc, myround(1 - acc)))
    print()
    print(classification_report(y_true, y_pred))


def plot_knn(data, grid, y_true, y_pred):
    X_plot = data['X_train'].values
    y_plot = data['y_train'].values.astype(np.integer)
    plot_decision_regions(X_plot, y_plot, clf=grid.best_estimator_, legend=2, scatter_kwargs=dict(s=20), markers='+o')

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plot_confusion_matrix(conf_mat=cm, show_absolute=True, show_normed=True, colorbar=True)
