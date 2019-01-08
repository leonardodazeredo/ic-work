import numpy as np
import random
from copy import copy
from pprint import pprint


def generate_points(n):
    '''
    Função que gera e retorna n points, uniformemente distribuídos no quadrado formado
    pelos points (-1, 1), (-1, -1), (1, 1) e (1, -1).

    O resultado é uma lista tipo array no formato [(x1, y1), (x2, y2), ..., (xn, yn)].
    '''
    xs = np.random.uniform(-1, 1, n)
    ys = np.random.uniform(-1, 1, n)
    return list(zip(xs, ys))


def datapoint_to_x_column_vector(datapoint):
    '''
    Função que transforma um único datapoint do formato ([x, y], rótulo) para um vetor
    coluna com as features do datapoint.

    Adicionalmente, adicionado o valor fixo 1, correnpondente ao peso w0 do vetor de
    pesos w que representa o bias.
    '''
    point = datapoint[0]
    return np.matrix([1] + list(point)).transpose()


def make_X_matrix_and_y_vector(dataset):
    '''
    Produz o par de matriz X de features e o vetor coluna y de labels a partir
    de uma amostra.
    '''
    X = list()
    y = list()

    for point, label in dataset:
        X.append([1] + list(point))
        y.append(label)

    X = np.matrix(X)
    y = np.matrix(y).transpose()

    return X, y


def generate_dataset(f, n, transform_function=None, noise_function=None):
    '''
    Generates a dataset of size n as a list of random data points,
    each one with the form ([x, y], label), where 'label' is the evaluation
    of f at the point [x, y].
    Gera uma nova lista e não altera a passada como parâmetro.
    '''
    points = generate_points(n)
    dataset = [(p, f(p)) for p in points]

    # Adicionando ruído, caso desejado.
    if noise_function is not None:
        noise_function(dataset)

    # Realizando uma tranformação, caso desejado.
    if transform_function is not None:
        dataset = transform_function(dataset)

    return dataset


def add_noise_10per(dataset):
    '''
    Adiciona ruído artificial a uma amostra mudando o rótulo de 10 por cento
    de seus elementos, selecionados aleatóriamente.
    Altera a lista passada como parâmetro.
    '''
    indexes = list(range(len(dataset)))
    size = int(len(dataset) * 0.1)
    indexes = random.sample(indexes, size)

    for idx in indexes:
        point, label = dataset[idx]
        dataset[idx] = (point, label * -1)


def transform_datapoint_features_to_x_y_xy_xx_yy(datapoint):
    '''
    Aplica a tranformação (x1, x2) -> (x1 , x2 , x1 * x2 , x1^2 , x2^2 )
    a um ponto.
    '''
    point, label = datapoint
    x, y = point
    return ((x, y, x * y, x**2, y**2), label)


def transform_dataset_features_to_x_y_xy_xx_yy(dataset):
    '''
    Aplica a tranformação (x1, x2) -> (x1 , x2 , x1 * x2 , x1^2 , x2^2 )
    a todos os pontos de uma amostra.
    '''
    dts = copy(dataset)
    for i, datapoint in enumerate(dts):
        dts[i] = transform_datapoint_features_to_x_y_xy_xx_yy(datapoint)
    return dts
