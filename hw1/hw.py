import numpy as np
from tqdm import tqdm
import random
import multiprocessing


def gerar_pontos(n):
    '''
    Funcao que gera e retorna n pontos uniformemente distribuidos no quadrado formado
    pelos pontos (-1, 1), (-1, -1), (1, 1) e (1, -1).

    O resultado e uma lista tipo array no formato [(x1, y1), (x2, y2), ..., (xn, yn)].
    '''
    xs = np.random.uniform(-1, 1, n)
    ys = np.random.uniform(-1, 1, n)
    return list(zip(xs, ys))


def datapoint_to_x_column_matrix(datapoint):
    '''
    Funcao que transforma um unico datapoint do formato ([x, y], rotulo) para um vetor
    coluna com as features do datapoint.

    Adicionalmente, adicionado o valor fixo 1, correnpondente ao peso w0 do vetor de
    pesos w que representa o bias.
    '''
    x, y = datapoint[0]
    return np.matrix([1, x, y]).transpose()


def sign(s):
    '''
    Funcao de sinal para a conversao do valor do sinal produzido pelo porceptron na
    label correspondente.
    '''
    if s > 0:
        return 1
    elif s < 0:
        return -1
    return 0


def h(w, x):
    '''
    Definicao de uma hipotese h, funcao de um datapoint, parametrizada pelo seu vetor
    de pesos w.
    '''
    return sign(w.transpose() * datapoint_to_x_column_matrix(x))


def pla(dataset):
    '''
    Funcao que executa o algoritimo de aprendizado do perceptron em um dataset, que
    consiste de uma lista com elementos do tipo 'datapoint'.
    O dataset suposto ser linearmente separavel.

    O valor de retorno inclui o vetor de pesos w final e o numero de interecoes
    executadas.
    '''

    # Criando e iniciando o vetor coluna de pesos w com valores 0.
    w = np.matrix([0, 0, 0]).transpose()

    # Contador de iteracoes realizadas
    n_iter = 0

    # Loop infinito.
    while True:
        # Avaliando a classe indicada pela hipotese atual
        # (ou seja, definida com o w atual) para todos os datapoints do dataset
        # e gerando uma lista com os datapoints que foram erroneamente classificados.
        missc_datapoints = [dp for dp in dataset if h(w, dp) != dp[1]]

        # Caso a lista de pontos erroneamente classificados esteja vazia,
        # finaliza o loop.
        if not missc_datapoints:
            break

        # Um datapoint erroneamente classificado selecionado aleatoriamente.
        datapoint = random.choice(missc_datapoints)

        # Aplicando a regra de atualizacao dos pesos w com base no ponto selecionado.
        w = w + datapoint[1] * datapoint_to_x_column_matrix(datapoint)

        n_iter += 1

    return dict(w=w, n_iter=n_iter)


def estimativa_de_probabilidade(f, w):
    '''
    Funcao auxiliar que estima a Probabilidade de uma funcao f concordar com uma funcao
    h, parametrizada por w.
    '''
    rodadas_n = 100
    pontos = gerar_pontos(rodadas_n)

    concordacia_count = 0

    for p in pontos:
        if f(p) == h(w, (p, None)):
            concordacia_count += 1

    return concordacia_count / rodadas_n


def rodar_experimento(n):
    '''
    Funcao que executa um unico experimento completo com todas as etapas.
    Nomeadamente:

    a. geracao de uma funcao f, como a funcao de reta que liga dois pontos gerados
    aleatoriamente.

    b. geracao de uma amostra de pontos de tamanho n e sua classificacao segundo f.

    c. montagem de uma lista correspondente ao dataset, com elementos do tipo datapoint:
    ([x, y], rotulo)

    d. invocacao do pla sobre o dataset gerado.

    e. avaliacao da estimativa da probabilidade entre a hipotese selecionada e a
    funcao alvo.

    O retorno inclui o numero de iteracoes ate a convergencia e a estimativa da
    probabilidade de concordacia.
    '''

    # Gerando 2 pontos aleatorios para definir uma reta.
    pontos = gerar_pontos(2)
    (x1, y1) = pontos[0]
    (x2, y2) = pontos[1]

    def reta(x):
        '''
        Definicao da funcao de reta que intercepta os pontos acima gerados.
        '''
        m = (y1 - y2) / (x1 - x2)
        d = (x1 * y1 - x2 * y1) / (x1 - x2)
        return m * x + d

    def f(ponto):
        '''
        Funcao alvo de classificao dos pontos definida a partir da funcao de reta.
        '''
        x, y = ponto
        if y > reta(x):
            return 1
        elif y < reta(x):
            return -1
        else:
            return 0

    # Gerando dataset
    xv = gerar_pontos(n)
    dataset = [(x, f(x)) for x in xv]

    resultado = pla(dataset)

    probabilidade_estimada = estimativa_de_probabilidade(f, resultado['w'])

    return dict(n_iter=resultado['n_iter'], prob=probabilidade_estimada)


def rodar_multiplos_experimentos(n, n_experimentos=1000):
    '''
    Funcao que executa n_experimentos experimentos independentes, cada qual com sua
    respectiva amostra gerada independentemente com tamanho n.
    '''

    print("---\nN =", n)

    # A linhas abaixo executam n_experimentos vezes a funcao rodar_experimento(n),
    # de forma paralela e indepentende.
    resultados = list()
    n_l = [n for _ in range(n_experimentos)]
    pool = multiprocessing.Pool()
    for r in tqdm(pool.imap_unordered(rodar_experimento, n_l), total=n_experimentos):
        resultados.append(r)
    pool.close()
    pool.join()

    # Produzindo uma lista contendo apenas os valores dos numeros de iteracoes
    # encontrado em cada experimento.
    total_iteracoes = sum(map(lambda a: a['n_iter'], resultados))

    # Produzindo uma lista contendo apenas os valores das extimativas de
    # probabilidade de concordancia encontradas em cada experimento.
    total_probs = sum(map(lambda a: a['prob'], resultados))

    # Calculando o numero medio de iteracoes
    print("\nIteracoes:               {}".format(total_iteracoes / n_experimentos))

    # Calculando o valor medio da probabilidade de concordancia.
    print("Probabilidade de erro:   {}\n".format(round(1 - (total_probs / n_experimentos), 4)))


if __name__ == '__main__':
    rodar_multiplos_experimentos(10)
    rodar_multiplos_experimentos(100)
    rodar_multiplos_experimentos(200)
