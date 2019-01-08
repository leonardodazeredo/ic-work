import pandas as pd
from pprint import pprint
from sklearn.model_selection import train_test_split


def split_data():
    bananadata = pd.read_csv("./banana/banana.csv")
    bananadata_train, bananadata_test = train_test_split(bananadata, test_size=0.20)

    bananadata_train.to_csv("./banana/bananadata_train.csv", sep='\t')
    bananadata_test.to_csv("./banana/bananadata_test.csv", sep='\t')


if __name__ == '__main__':
    split_data()
else:
    bananadata_train = pd.read_csv("./banana/bananadata_train.csv", sep='\t').drop('I', axis=1)
    print(bananadata_train.shape)
    X_train = bananadata_train.drop('Class', axis=1)
    y_train = bananadata_train['Class']

    bananadata_test = pd.read_csv("./banana/bananadata_test.csv", sep='\t').drop('I', axis=1)
    print(bananadata_test.shape)
    X_test = bananadata_test.drop('Class', axis=1)
    y_test = bananadata_test['Class']

    bananadata = pd.read_csv("./banana/banana.csv")
    X = bananadata.drop('Class', axis=1)
    y = bananadata['Class']

    data = dict(
        bananadata=bananadata,
        X=X,
        y=y,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )
