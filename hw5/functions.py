from datasets import datapoint_to_x_column_vector, generate_points
from math import exp, log
import math


def euclidean_distance(a, b):
    '''
    Calcula a distancia euclidiana de dois pontos
    '''
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def E_surface(u, v):
    '''
    Superficie e erro E(u, v) = (u*e^(v) - 2*v*e^(-u))^2
    '''
    return (u * (math.e**v) - 2 * v * (math.e**(-u)))**2


def derp_u(u, v):
    '''
    Calculando a derivada parcial de E(u, v) = (u*e^(v) - 2*v*e^(-u))^2 com respeito a u
    '''
    return 2 * (math.e**(v) + 2 * v * math.e**(-u)) * (u * math.e**(v) - 2 * v * math.e**(-u))


def derp_v(u, v):
    '''
    Calculando a derivada parcial de E(u, v) = (u*e^(v) - 2*v*e^(-u))^2 com respeito a v
    '''
    return 2 * (u * math.e**(v) - 2 * math.e**(-u)) * (u * math.e**(v) - 2 * v * math.e**(-u))


def gradient(w, datapoint):
    '''
    Calcula o gradiente do erro em relação a um único datapoint
    '''
    point, label = datapoint
    x = datapoint_to_x_column_vector(datapoint)
    return (-label * x) / (1 + exp(label * (w.transpose() * x)))


def cross_entropy_error(w, dataset):
    '''
    Calcula o erro de entropia cruzada de w com todo o dataset
    '''
    sm = 0
    for datapoint in dataset:
        point, label = datapoint
        sm += log(1 + exp(-label * (w.transpose() * datapoint_to_x_column_vector(datapoint))))
    return sm / len(dataset)


def sigmoid(s):
    '''
    Função para a conversão do valor do sinal na probabilidade.
    '''
    return exp(s) / (1 + exp(s))


def h(w, datapoint):
    '''
    Definiçao de uma hipótese h, função de um datapoint, parametrizada pelo seu vetor
    de pesos w.
    '''
    return sigmoid(w.transpose() * datapoint_to_x_column_vector(datapoint))


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
