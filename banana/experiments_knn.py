import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.model_selection import GridSearchCV
from analysis import plot_knn, print_results_knn
from sklearn.neighbors import KNeighborsClassifier

from dataset import data

# param_grid = {'weights': ['distance'],
#             'n_neighbors': list(range(5, 55, 5))}

param_grid = {'weights': ['distance'],
            'n_neighbors': [35]}

k_fold = 10


def run():
    grid = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, scoring="accuracy", return_train_score=False, verbose=1, n_jobs=-1, cv=k_fold)
    grid.fit(data['X_train'], data['y_train'])

    y_true, y_pred = data['y_test'], grid.predict(data['X_test'])

    print_results_knn(data, grid, y_true, y_pred)

    plot_knn(data, grid, y_true, y_pred)


if __name__ == '__main__':
    run()
    plt.show()
