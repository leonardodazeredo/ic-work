from utils.parallel import map_in_parallel
from utils.misc import estimation_value, get_option
from datasets import generate_dataset
from functions import generate_line_function, generate_f, derp_u, derp_v, E_surface, euclidean_distance
from learning_algorithms import logir_sgd, evaluate_E_in, estimate_E_out


def run_experiment_A():
    '''
    Executa o experimento referente às questões (1), (2)
    '''
    l_rate = 0.1

    # vetor inicial
    w = (1.0, 1.0)

    it = 1

    while True:
        # calculando valores para o vetor w(i)
        u, v = (w[0] - l_rate * derp_u(w[0], w[1]), w[1] - l_rate * derp_v(w[0], w[1]))

        # valor na superfície de erro do experimento
        E = E_surface(u, v)

        print("#{}: ({}, {}), E={}".format(it, u, v, E))

        # atualizando o estado atual do vetor
        w = u, v

        # interrompendo loop segundo critério do experimento
        if E < 10**(-14):
            break

        it += 1

    print("\nE value at last iteration:   {}".format(E))

    # testando as opções da questão (1)
    option = get_option(it, [1, 3, 5, 10, 17])
    print("Last iteration:              {} {}\n".format(it, option))

    # testando as opções da questão (2)
    opts = [(1.0, 1.0), (0.713, 0.045), (0.016, 0.112), (-0.083, 0.029), (0.045, 0.024)]
    option = get_option(w, opts, euclidean_distance)
    u, v = w
    print("Point at last iteration:     ({}, {}) {}\n".format(round(u, 3), round(v, 3), option))


def run_experiment_B():
    '''
    Executa o experimento referente às questões (3)
    '''
    l_rate = 0.1

    # vetor inicial
    w = (1.0, 1.0)

    for i in range(1, 16):
        # atualizando w com o movimento na primeira coordenada
        u = w[0] - l_rate * derp_u(w[0], w[1])
        w = u, w[1]

        # atualizando w com o movimento na segunda coordenada
        v = w[1] - l_rate * derp_v(w[0], w[1])
        w = w[0], v

        u, v = w
        # calculando erro
        E = E_surface(u, v)
        print("#{}: ({}, {}), E={}".format(i, u, v, E))

    # testando as opções da questão (3)
    option = get_option(E, [10**(-1), 10**(-7), 10**(-14), 10**(-17), 10**(-20)])
    print("\nE value at last iteration:   {} {}\n".format(round(E, 6), option))


def run_experiment_C(n):
    '''
    Executa o experimento referente às questões (4) e (5)
    '''
    # Gerando uma reta aleatória e a função de avaliação corespondente.
    line_function = generate_line_function()
    pseudo_f = generate_f(line_function)

    # Gerando amostra sem ruído.
    dataset = generate_dataset(pseudo_f, n)

    # Executando regressão.
    result = logir_sgd(dataset)
    w = result['w']
    epochs = result['epochs']

    # Avaliando resultados de logir_sgd.
    E_in = evaluate_E_in(w, dataset)
    estimated_E_out = estimate_E_out(pseudo_f, w)

    return dict(w=w, E_in=E_in, estimated_E_out=estimated_E_out, dataset=dataset, epochs=epochs)


def run_multiple_experiments_C(n=100, n_experiments=100):
    '''
        Executa o experimento definido na função run_experiment_C(n) n_experiments vezes
    a fim de produzir resultados estáveis via método de Monte Carlo.
        Descrição geral: executa run_experiment_C(n) n_experiments vezes, coleta os
    resultados de cada execução e imprime a avaliação do Ein, a estimativa de Eout e
    o número de epochs.
    '''
    print("-----------------------------------------------------------------------------")
    print("N =", n)

    # Executando n_experiments vezes a função run_experiment_C(n), de forma paralela e
    # indepentende.
    list_of_n = [n for _ in range(n_experiments)]
    results = map_in_parallel(run_experiment_C, list_of_n, with_progress_bar=True)

    # Extraindo os valores de Ein de cada experimento e somando.
    E_ins_sum = sum(map(lambda a: a['E_in'], results))

    # Extraindo as estimativas de Eout de cada experimento e somando.
    estimated_E_outs_sum = sum(map(lambda a: a['estimated_E_out'], results))

    # Extraindo o número de epochs de cada experimento e somando.
    epochs_iterations = sum(map(lambda a: a['epochs'], results))

    # Calculando a média dos valores de Ein
    E_ins_average = estimation_value(E_ins_sum, n_experiments)
    print("Eins average:              {}".format(E_ins_average))

    # Calculando o valor médio das estimativas de Eout e comparando com as opções
    E_outs_average = estimation_value(estimated_E_outs_sum, n_experiments)
    option = get_option(E_outs_average, [0.025, 0.050, 0.075, 0.100, 0.125])
    print("Estimated Eouts average:   {}".format(E_outs_average), option)

    # Calculando o número médio de epochs e comparando com as opções
    epochs_average = estimation_value(epochs_iterations, n_experiments)
    option = get_option(epochs_average, [350, 550, 750, 950, 1750])
    print("Epochs average:            {}".format(epochs_average), option)


if __name__ == '__main__':
    run_experiment_A()
    run_experiment_B()
    run_multiple_experiments_C()
