import numpy as np
from utils.parallel import map_in_parallel
from utils.misc import estimation_value, get_option, to_letter
from pprint import pprint
from datasets import generate_dataset, add_noise_10per, transform_dataset_features_to_x_y_xy_xx_yy
from functions import f_circum, generate_line_function, generate_f
from learning_algorithms import lr, aux_pla, evaluate_E_in, estimate_E_out, estimate_agreement_probability
import random


def run_experiment_D(results):
    '''
    Esta função executa o experimento de comparação referente às opções da questão (5).
    Descrição geral: recebe uma lista de resultados de múltiplos experimentos, extrái os
    vetores de pesos w, amostra um subconjunto destes com 10 elementos, estima a
    probabilidade de concordância entre a hipótese definida por cada um deles e cada uma
    das opções da questão (5).
    '''
    # Extraindo uma amostra de 10 w's dos resultados.
    w_learned_list = list(map(lambda e: e['w'], random.sample(results, 10)))

    # Declarando lista de w's correspondentes às opções da questão (5)
    w_to_compare_list = [
        [-1, -0.05, 0.08, 0.13, 1.5, 1.5],
        [-1, -0.05, 0.08, 0.13, 1.5, 15],
        [-1, -0.05, 0.08, 0.13, 15, 1.5],
        [-1, -1.5, 0.08, 0.13, 0.05, 0.05],
        [-1, -0.05, 0.08, 1.5, 0.15, 0.15]
    ]

    # Convertendo o formato dos valores de listas python para matrizes numpy.
    w_to_compare_list = list(map(lambda e: np.matrix(e).transpose(), w_to_compare_list))

    p_l = []
    for i, wi in enumerate(w_to_compare_list):
        # Realizando a estimativa de probabilidade de concordância entre cada opção e o valor de w amostrando em w_learned_list.
        s_sum = 0
        for wj in w_learned_list:
            s_sum += estimate_agreement_probability(f_circum, wj, wi, transform_function=transform_dataset_features_to_x_y_xy_xx_yy)
        p_l.append(estimation_value(s_sum, len(w_learned_list)))

    # Imprimindo resultados
    print(p_l, to_letter(p_l.index(max(p_l))))


def run_experiment_C(n):
    '''
    Esta função executa um experimento único referente à questão (6).
    Descrição geral: gera uma amostra de tamanho N, com ruído, de uma função alvo
    definida pela função f_circum, aplica a transformação não linear (), executa a regressão linear sobre ela
    e avalia o erro dentro da amostra, assim como uma estimativa do erro fora da
    amostra.
    '''
    # Utilizando a função alvo definida para as questões (4) até (6): f_circum(x1, x2) = sign(x1^2 + x2^2 − 0.6)
    f = f_circum

    # Gerando uma amostra a partir da função alvo, aplicadas a tranformação não linear e a adição de ruído.
    dataset_transformed = generate_dataset(f, n, transform_function=transform_dataset_features_to_x_y_xy_xx_yy, noise_function=add_noise_10per)

    # Executando regressão.
    w_ = lr(dataset_transformed)['w']

    # Avaliando resultados de w.
    E_in = evaluate_E_in(w_, dataset_transformed)
    estimated_E_out = estimate_E_out(f, w_, transform_function=transform_dataset_features_to_x_y_xy_xx_yy, noise_function=add_noise_10per)

    return dict(w=w_, E_in=E_in, estimated_E_out=estimated_E_out, dataset=dataset_transformed)


def run_multiple_experiments_C_and_D(n, n_experiments=1000):
    '''
        Exetuta o experimento definido na função run_experiment_C(n) n_experiments vezes
    a fim de produzir resultados estáveis via método de Monte Carlo.
        Adicionalmente, executa o experimento definido por run_experiment_D(results) sobre
    a lista de resultados produzida por run_experiment_C(n).
        Descrição geral: executa run_experiment_C(n) n_experiments vezes, coleta os
    resultados de cada execução e imprime a avaliação do Ein e a estimativa de Eout.
    '''
    print("-----------------------------------------------------------------------------")
    print("N =", n)

    # Executando n_experiments vezes a função run_experiment_A(n), de forma paralela e
    # indepentende.
    list_of_n = [n for _ in range(n_experiments)]
    results = map_in_parallel(run_experiment_C, list_of_n, with_progress_bar=True)

    # Executando comparação e imprimindo resultados para a questão (5)
    run_experiment_D(results)

    # Extraindo os valores de Ein de cada experimento e somando.
    E_ins_sum = sum(map(lambda a: a['E_in'], results))

    # Extraindo as estimativas de Eout de cada experimento e somando.
    estimated_E_outs_sum = sum(map(lambda a: a['estimated_E_out'], results))

    # Calculando a média dos valores de Ein
    E_ins_average = estimation_value(E_ins_sum, n_experiments)
    print("Eins average:              {}".format(E_ins_average))

    # Calculando o valor médio das estimativas de Eout.
    E_outs_average = estimation_value(estimated_E_outs_sum, n_experiments)
    option = get_option(E_outs_average, [0, 0.1, 0.3, 0.5, 0.8])
    print("Estimated Eouts average:   {}".format(E_outs_average), option)


def run_experiment_B(n):
    '''
    Esta função executa um experimento único referente à questão (4).
    Descrição geral: gera uma amostra de tamanho N, com ruído, de uma função alvo
    definida pela função f_circum, executa a regressão linear sobre ela
    e avalia o erro dentro da amostra, assim como uma estimativa do erro fora da
    amostra.
    '''
    # Utilizando a função alvo definida para as questões (4) até (6): f_circum(x1, x2) = sign(x1^2 + x2^2 − 0.6)
    f = f_circum

    # Gerando uma amostra de f de tamanho n e adicionando ruído.
    dataset = generate_dataset(f, n, noise_function=add_noise_10per)

    # Executando regressão.
    w = lr(dataset)['w']

    # Avaliando resultados de w.
    E_in = evaluate_E_in(w, dataset)
    estimated_E_out = estimate_E_out(f, w)

    return dict(w=w, E_in=E_in, estimated_E_out=estimated_E_out, dataset=dataset)


def run_multiple_experiments_B(n, n_experiments=1000):
    '''
        Exetuta o experimento definido na função run_experiment_B(n) n_experiments vezes
    a fim de produzir resultados estáveis via método de Monte Carlo.
        Descrição geral: executa run_experiment_B(n) n_experiments vezes, coleta os
    resultados de cada execução e imprime a avaliação do Ein e a estimativa de Eout.
    '''
    print("-----------------------------------------------------------------------------")
    print("N =", n)

    # Executando n_experiments vezes a função run_experiment_A(n), de forma paralela e
    # indepentende.
    list_of_n = [n for _ in range(n_experiments)]
    results = map_in_parallel(run_experiment_B, list_of_n, with_progress_bar=True)

    # Extraindo os valores de Ein de cada experimento e somando.
    E_ins_sum = sum(map(lambda a: a['E_in'], results))

    # Extraindo as estimativas de Eout de cada experimento e somando.
    estimated_E_outs_sum = sum(map(lambda a: a['estimated_E_out'], results))

    # Calculando a média dos valores de Ein
    E_ins_average = estimation_value(E_ins_sum, n_experiments)
    option = get_option(E_ins_average, [0, 0.1, 0.3, 0.5, 0.8])
    print("Eins average:              {}".format(E_ins_average), option)

    # Calculando o valor médio das estimativas de Eout.
    E_outs_average = estimation_value(estimated_E_outs_sum, n_experiments)
    print("Estimated Eouts average:   {}".format(E_outs_average))


def run_experiment_A(n):
    '''
    Esta função executa um experimento único referente às questões (1), (2) e (3).
    Descrição geral: gera uma amostra de tamanho N, sem ruído, de uma função alvo
    baseada em uma reta gerada aleatoriamente, executa a regressão linear sobre ela
    e avalia o erro dentro da amostra, assim como uma estimativa do erro fora da
    amostra.
    '''
    # Gerando uma reta aleatória e a função alvo corespondente.
    line_function = generate_line_function()
    f = generate_f(line_function)

    # Gerando amostra sem ruído.
    dataset = generate_dataset(f, n)

    # Executando regressão.
    w = lr(dataset)['w']

    # Avaliando resultados de w.
    E_in = evaluate_E_in(w, dataset)
    estimated_E_out = estimate_E_out(f, w)

    return dict(w=w, E_in=E_in, estimated_E_out=estimated_E_out, dataset=dataset)


def run_multiple_experiments_A(n, n_experiments=1000, run_pla=False):
    '''
        Exetuta o experimento definido na função run_experiment_A(n) n_experiments vezes
    a fim de produzir resultados estáveis via método de Monte Carlo.
        Adicionalmente, se run_pla=True, executa o algoritmo PLA para cada par de amostra
    e vetor w produzidos pela regressão linear, calculando o número de iterações médio até
    a convergencia quando o PLA inicia em w.
        Descrição geral: executa run_experiment_A(n) n_experiments vezes, coleta os
    resultados de cada execução; com run_pla=False, imprime Ein e Eout; com run_pla=True,
    executa o PLA iniciado com cada w produzido pela regressão.
    '''
    print("-----------------------------------------------------------------------------")
    print("N =", n)

    # Executando n_experiments vezes a função run_experiment_A(n), de forma paralela e
    # indepentende.
    list_of_n = [n for _ in range(n_experiments)]
    results = map_in_parallel(run_experiment_A, list_of_n, with_progress_bar=True)

    if not run_pla:
        # Extraindo os valores de Ein de cada experimento e somando.
        E_ins_sum = sum(map(lambda a: a['E_in'], results))

        # Extraindo as estimativas de Eout de cada experimento e somando.
        estimated_E_outs_sum = sum(map(lambda a: a['estimated_E_out'], results))

        # Calculando o valor médio de Ein.
        E_ins_average = estimation_value(E_ins_sum, n_experiments)
        option = get_option(E_ins_average, [0, 0.001, 0.01, 0.1, 0.5])
        print("Eins average:              {}".format(E_ins_average), option)

        # Calculando o valor médio da probabilidade de concordância.
        E_outs_average = estimation_value(estimated_E_outs_sum, n_experiments)
        option = get_option(E_outs_average, [0, 0.001, 0.01, 0.1, 0.5])
        print("Estimated Eouts average:   {}".format(E_outs_average), option)

    if run_pla:
        list_of_dataset_w_pair = [(r['dataset'], r['w']) for r in results]

        # Executando o PLA para cara w, em paralelo.
        results = map_in_parallel(aux_pla, list_of_dataset_w_pair)

        # Extraindo o número de iterações de cada experimento e somando.
        total_iterations = sum(map(lambda a: a['n_iter'], results))

        # Calculando o número médio de iterações do PLA
        pla_iterations_average = estimation_value(total_iterations, n_experiments)
        option = get_option(pla_iterations_average, [1, 15, 300, 5000, 10000])
        print("PLA iterations average:    {}".format(pla_iterations_average), option)


if __name__ == '__main__':
    run_multiple_experiments_A(n=100)  # Questão (1) e (2)
    run_multiple_experiments_A(n=10, run_pla=True)  # Questão (3)
    run_multiple_experiments_B(n=1000)  # Questão (4)
    run_multiple_experiments_C_and_D(n=1000)  # Questão (5) e (6)
