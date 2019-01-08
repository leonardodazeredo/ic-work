import numpy as np
import random
from pprint import pprint
from utils.misc import estimation_value
from datasets import generate_dataset, make_X_matrix_and_y_vector, datapoint_to_x_column_vector
from functions import h


def estimate_E_out(f, w, n_rounds=1000, transform_function=None, noise_function=None):
    '''
    Estima o erro fora da amostra de h[w](x), gerando uma amostra (com ou sem ruído, com ou sem tranformação)
    e calculando sua média de concordância com f.
    '''
    dataset = generate_dataset(f, n_rounds, transform_function=transform_function, noise_function=noise_function)

    agreement_count = 0

    for datapoint in dataset:
        point, label = datapoint
        if label == h(w, datapoint):
            agreement_count += 1

    return 1 - estimation_value(agreement_count, n_rounds)


def estimate_agreement_probability(f, w1, w2, n_rounds=1000, transform_function=None, noise_function=None):
    '''
    Estima a probabilidadede concordância entre duas hióteses correspondentes a w1 e w2,
    gerando uma amostra (com ou sem ruído, com ou sem tranformação)
    e calculando a média de concordância entre elas.
    '''
    dataset = generate_dataset(f, n_rounds, transform_function=transform_function, noise_function=noise_function)

    agreement_count = 0

    for datapoint in dataset:
        if h(w1, datapoint) == h(w2, datapoint):
            agreement_count += 1

    return estimation_value(agreement_count, n_rounds)


def evaluate_E_in(w, dataset):
    '''
    Calcula o erro dentro da amostra (rotulada) dataset, de uma hipótese parametrizada por w.
    '''
    agreement_count = 0

    for datapoint in dataset:
        point, label = datapoint
        if label == h(w, datapoint):
            agreement_count += 1

    return 1 - estimation_value(agreement_count, len(dataset))


def lr(dataset):
    '''
    Executa o algoritimo de aprendizado em um passo da regressão linear, sobre uma amostra rotulada.
    '''
    # Extraindo a matriz X e o vetor y da amostra.
    X, y = make_X_matrix_and_y_vector(dataset)

    # Calculando a pseudo inversa de X.
    pseudo_inverse_of_X = (X.transpose() * X).I * X.transpose()

    # Determinando w analiticamente.
    w = pseudo_inverse_of_X * y

    return dict(w=w)


def aux_pla(arg):
    '''
    Função auxiliar para permitir a execução da função pla() em paralelo.
    '''
    return pla(arg[0], arg[1])


def pla(dataset, init_w=None):
    '''
    Função que executa o algoritmo de aprendizado do perceptron sobre uma amostra, que
    consiste de uma lista com elementos do tipo 'datapoint'.
    O dataset é aqui suposto ser linearmente separável.

    O valor de retorno inclui o vetor de pesos w final e o número de intereções
    executadas.

    Se init_w for passado à função, este é utilizado como w incial.
    '''
    if init_w is None:
        # Criando e iniciando o vetor coluna de pesos w com valores 0.
        w = np.matrix([0] + [0 for _ in dataset[0]]).transpose()
    else:
        w = init_w

    # Contador de iterações realizadas
    n_iter = 0

    # Loop infinito.
    while True:
        # Avaliando a classe indicada pela hipótese correspondente ao w da iteração corrente
        # em todos os datapoints do dataset (amostra) e gerando uma lista com os datapoints
        # que foram erroneamente classificados.
        missc_datapoints = [dp for dp in dataset if h(w, dp) != dp[1]]

        # Caso a lista de datapoints erroneamente classificados esteja vazia,
        # finaliza o loop.
        if not missc_datapoints:
            break

        # Um datapoint erroneamente classificado é selecionado aleatoriamente.
        datapoint = random.choice(missc_datapoints)

        # Aplicando a regra de atualização dos pesos w com base no point selecionado.
        w = w + datapoint[1] * datapoint_to_x_column_vector(datapoint)

        n_iter += 1

    return dict(w=w, n_iter=n_iter)
