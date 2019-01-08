from pprint import pprint
from datasets import datapoint_to_x_column_vector, generate_points


def sign(s):
    '''
    Função de sinal para a conversão do valor do sinal na label correspondente.
    '''
    if s > 0:
        return 1
    elif s < 0:
        return -1
    return 0


def h(w, datapoint):
    '''
    Definiçao de uma hipótese h, função de um datapoint, parametrizada pelo seu vetor
    de pesos w.
    '''
    return sign(w.transpose() * datapoint_to_x_column_vector(datapoint))


def generate_line_function():
    '''
    Gera uma função de reta aleatória.
    '''
    # Gerando 2 pontos aleatórios para definir uma line_function.
    points = generate_points(2)
    (x1, y1) = points[0]
    (x2, y2) = points[1]

    def line_function(x):
        '''
        Definição da função de line_function que intercepta os points acima gerados.
        '''
        m = (y1 - y2) / (x1 - x2)
        d = (x1 * y1 - x2 * y1) / (x1 - x2)
        return m * x + d

    return line_function


def generate_f(line_function):
    def f(point):
        '''
        Função alvo de classifição dos pontos definida a partir da função de line_function.
        '''
        x, y = point
        if y > line_function(x):
            return 1
        elif y < line_function(x):
            return -1
        else:
            return 0

    return f


def f_circum(point):
    '''
    Função alvo das questões (4), (5) e (6)
    '''
    def undo_transform(p):
        '''
        Desfaz a tranformação não linear.
        '''
        return (p[0], p[1])

    x1, x2 = undo_transform(point)
    return sign(x1**2 + x2**2 - 0.6)
