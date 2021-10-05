# to manipulate files and data structures
from os import listdir
from os.path import isfile, join
import numpy as np
from sys import maxsize

# solvers
import pulp as pl  # (https://github.com/coin-or/pulp) solves CBC and GUROBI algorithm
from munkres import Munkres  # (https://software.clapper.org/munkres/) solves hungarian algorithm

# to run solvers in parallel
from multiprocessing import Process

# to measure elapsed time in each instance
import time


# constants
BASE_PATH_A = './instancias/insta'
BASE_PATH_B = './instancias/instb'
BASE_PATH_RESULTS = './results'


def write_output(file_name, data):
    [_, values] = list(zip(*data))
    np.savetxt(f'{BASE_PATH_RESULTS}/{file_name}.csv', np.array(values))


def list_files(path):
    files = [f'{path}/{f}' for f in listdir(path) if isfile(join(path, f))]
    files.sort()
    return files


def generate_final_result_by_type(solvers_name, type, path):
    files = [name.split('/').pop() for name in list_files(path)]
    results_by_solver = [np.genfromtxt(f'{BASE_PATH_RESULTS}/{solver}-{type}.csv').tolist() for solver in solvers_name]

    header = ['file_name'] + [solver for (solver) in solvers_name]
    data_rows = [[f"{value:.4f}" for value in row] for row in list(zip(*results_by_solver))]

    with open(f'{BASE_PATH_RESULTS}/FINAL-{type}.csv', 'w') as output:
        output.write(','.join(header) + '\n')
        for i in range(len(files)):
            output.write(','.join([files[i], *data_rows[i]]) + '\n')


def generate_final_results(solvers_name):
    generate_final_result_by_type(solvers_name, 'A', BASE_PATH_A)
    generate_final_result_by_type(solvers_name, 'B', BASE_PATH_B)


def solve_pulp(file, profit_matrix, solver_name):
    n = len(profit_matrix)
    prob = pl.LpProblem(file, pl.LpMaximize)
    solver = pl.getSolver(solver_name, timeLimit=30)

    # define as variaveis
    variables = pl.LpVariable.dicts("x", [(j, i) for i in range(n) for j in range(n)], lowBound=0, upBound=1,
                                    cat=pl.LpInteger)

    # define função objetivo, somatório (custo * variavel)
    prob += pl.lpSum([profit_matrix[i][j] * variables[(i, j)] for i in range(n) for j in range(n)])

    # define limitações
    for i in range(n):
        prob += pl.lpSum([variables[(i, j)] for j in range(n)]) == 1

    for j in range(n):
        prob += pl.lpSum([variables[(i, j)] for i in range(n)]) == 1

    start_time = time.perf_counter()
    # resolve
    status = prob.solve(solver=solver)
    elapsed_time = time.perf_counter() - start_time
    # Status
    # print("Status:", pl.LpStatus[status])

    # Valor das variaveis
    # for v in prob.variables():
    #     print(v.name, "=", v.varValue)
    #
    return elapsed_time


def solve_hungarian(_, profit_matrix):
    start_time = time.perf_counter()
    cost_matrix = []
    # O Munkres recebe uma matriz de custo, no nosso caso queremos o contrário, então precisamos transformar a matriz de lucro em uma de custo
    for row in profit_matrix:
        cost_row = []
        for col in row:
            cost_row += [maxsize - col]
        cost_matrix += [cost_row]

    m = Munkres()
    indexes = m.compute(cost_matrix)
    total = sum([profit_matrix[row][column] for row, column in indexes])
    elapsed_time = time.perf_counter() - start_time

    # print(f'total profit={total}')
    return elapsed_time


def solve(file, profit_matrix, solver_name):
    print("Starting ", file)
    if solver_name == 'HUNGARIAN':
        return solve_hungarian(file, profit_matrix)
    return solve_pulp(file, profit_matrix, solver_name)


def load_matrix(file_name, data_type):
    return np.loadtxt(file_name, dtype=data_type).tolist()


def load_files_matrix(files, data_type):
    return dict((file_name, load_matrix(file_name, data_type)) for file_name in files)


def run_a(solver_name):
    costs_a = load_files_matrix(list_files(BASE_PATH_A), 'i')

    # toy project
    # solution_time_a = [('./instancias/insta/toy_project.txt',
    #                     solve('./instancias/insta/toy_project.txt', costs_a['./instancias/insta/toy_project.txt'],
    #                           solver_name)), ('./instancias/insta/toy_project.txt',
    #                     solve('./instancias/insta/toy_project.txt', costs_a['./instancias/insta/toy_project.txt'],
    #                           solver_name))]
    # print(solution_time_a)

    # executa todas as instancias A e retorna o tempo de resposta na forma (arquivo, tempo)
    solution_time_a = [(key, solve(key, costs_a[key], solver_name)) for key in costs_a]
    write_output(f'{solver_name}-A', solution_time_a)


def run_b(solver_name):
    costs_b = load_files_matrix(list_files(BASE_PATH_B), 'f')
    solution_time_b = [(key, solve(key, costs_b[key], solver_name)) for key in costs_b]
    write_output(f'{solver_name}-B', solution_time_b)


def run_parallel(tasks):
    running_tasks = [Process(target=task, args=(args,)) for (task, args) in tasks]
    for running_task in running_tasks:
        running_task.start()
    for running_task in running_tasks:
        running_task.join()


def run_solvers(solvers):
    instances = [(run_a, solver) for solver in solvers] + [(run_b, solver) for solver in solvers]
    run_parallel(instances)


if __name__ == '__main__':
    solvers = ['PULP_CBC_CMD', 'GUROBI', 'HUNGARIAN']
    run_solvers(solvers)
    generate_final_results(solvers)
