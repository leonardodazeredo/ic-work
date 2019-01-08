import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from analysis import print_results_svm, plot_svm
from mlxtend.plotting import category_scatter

from dataset import data


# param_grid = {'gamma': [1, 0.5, 0.01], 'kernel': ['sigmoid']}
# param_grid = {'kernel': ['linear']}
# param_grid = {'kernel': ['poly'], 'degree': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
param_grid = {'kernel': ['rbf']}
# param_grid = {'gamma': [1], 'kernel': ['sigmoid']}


k_fold = 10


def run():
    grid = GridSearchCV(estimator=SVC(), param_grid=param_grid, scoring="accuracy", return_train_score=False, verbose=1, n_jobs=-1, cv=k_fold)
    grid.fit(data['X_train'], data['y_train'])

    y_true, y_pred = data['y_test'], grid.predict(data['X_test'])

    print_results_svm(data, grid, y_true, y_pred)

    plot_svm(data, grid, y_true, y_pred)


if __name__ == '__main__':
    # category_scatter(x='At1', y='At2', label_col='Class', data=bananadata, markers="+o")
    run()
    plt.show()
