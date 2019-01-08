import numpy as np
import random
from datasets import generate_dataset
from functions import cross_entropy_error, gradient
from utils.misc import myround


def estimate_E_out(f, w, n_rounds=1000):
    '''
    Estima o erro fora da amostra de h[w](x), gerando uma nova amostra
    e calculando o erro de entropia cruzada sobre ela.
    '''
    dataset = generate_dataset(f, n_rounds)
    return myround(cross_entropy_error(w, dataset))


def evaluate_E_in(w, dataset):
    '''
    Calcula o erro dentro da amostra (rotulada) dataset, de uma hipótese parametrizada por w.
    '''
    return myround(cross_entropy_error(w, dataset))


def logir_sgd(dataset, l_rate=0.01):
    '''
    Executa o algoritimo de aprendizado da regressão logística com gradiente descendente
    estocático, sobre uma amostra rotulada.
    '''
    # iniciando w com valores 0
    w = np.matrix([0] + [0 for _ in dataset[0]]).transpose()

    # iniciando a variável auxiliar w_old com valores 1, para entrar no loop abaixo
    # na primeira iteração
    w_old = np.matrix([1] + [1 for _ in dataset[0]]).transpose()

    epochs_count = 0

    # loop interrompido quando ||w^(t-1) - w^(t)|| < 0.01
    while np.linalg.norm(w_old - w) >= 0.01:

        w_old = w

        # gerando uma lista de indices embaralhados do dataset
        indexes = list(range(len(dataset)))
        random.shuffle(indexes)

        for idx in indexes:
            dtn = dataset[idx]

            w = w - l_rate * gradient(w, dtn)

        epochs_count += 1

    return dict(w=w, epochs=epochs_count)
