import numpy as np


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


def generate_dataset(f, n):
    '''
    Generates a dataset of size n as a list of random data points,
    each one with the form ([x, y], label), where 'label' is the evaluation
    of f at the point [x, y].
    Gera uma nova lista e não altera a passada como parâmetro.
    '''
    points = generate_points(n)
    dataset = [(p, f(p)) for p in points]
    return dataset
