# to manipulate files and data structures
from os import listdir
from os.path import isfile, join
import numpy as np

# solvers
import pulp as pl  # (https://github.com/coin-or/pulp) solves CBC and GUROBI algorithm

BASE_PATH = './instancias'
BASE_PATH_RESULTS = './results'
SOLVER = 'GUROBI'
THREE_MINUTES_IN_SECONDS = 3 * 60
PRESOLVE = -1  # Liga/Desliga o pré-processamento, default = -1: https://www.gurobi.com/documentation/9.5/refman/presolve.html
VAR_BRANCH = 3  # Determina a estratégia de escolha da variável ser ramificada, default = -1: https://www.gurobi.com/documentation/9.5/refman/varbranch.html
BRANCH_DIR = 0  # Determina qual nó vai escolhido pra ser explorado, default = 0: https://www.gurobi.com/documentation/9.5/refman/branchdir.html
CUTS = -1  # Determina a estrategia pra executar cortes, default = -1: https://www.gurobi.com/documentation/9.5/refman/cuts.html


def list_files(path):
    files = [f'{path}/{f}' for f in listdir(path) if isfile(join(path, f))]
    files.sort()
    return files


def write_solutions(solutions):
    header = ['Nome do Arquivo', 'Status', 'Nos Explorados', 'Limitante Dual', 'Melhor Resultado', 'Gap (%)']
    with open(f'{BASE_PATH_RESULTS}/T2.csv', 'w') as output:
        output.write(','.join(header) + '\n')
        for i in range(len(solutions)):
            output.write(','.join([str(value) for value in solutions[i]]) + '\n')


def solve(file_name, agents, tasks, profit_matrix, cost_matrix, capacity):
    prob = pl.LpProblem(file_name, pl.LpMaximize)
    solver = pl.getSolver(SOLVER, timeLimit=THREE_MINUTES_IN_SECONDS, Presolve=PRESOLVE, VarBranch=VAR_BRANCH,
                          Cuts=CUTS, BranchDir=BRANCH_DIR)

    # I = Agent, J = Task
    # define as variaveis
    variables = pl.LpVariable.dicts("x", [(i, j) for j in range(tasks) for i in range(agents)], lowBound=0, upBound=1,
                                    cat=pl.LpInteger)

    # define função objetivo, somatório (custo * variavel)
    prob += pl.lpSum([profit_matrix[i][j] * variables[(i, j)] for j in range(tasks) for i in range(agents)])

    # define limitações
    # Cada task só pode ser feita por um agente
    for j in range(tasks):
        prob += pl.lpSum([variables[(i, j)] for i in range(agents)]) == 1

    # Cada agente só pode ter no máximo tarefas que custem menor ou igual a sua capacidade
    for i in range(agents):
        prob += pl.lpSum([cost_matrix[i][j] * variables[(i, j)] for j in range(tasks)]) <= capacity[i]

    # Resolve
    status = prob.solve(solver=solver)

    gap = prob.solverModel.MIPGap * 100
    dual = prob.solverModel.ObjBound
    best_solution = prob.solverModel.ObjVal
    explored_nodes = prob.solverModel.NodeCount

    return [file_name, pl.LpStatus[status], explored_nodes, dual, best_solution, gap]


def read_file(filename):
    file = open(filename, 'r')
    numbers_by_lines = [line.split() for line in file.readlines()]
    numbers = [item for sublist in numbers_by_lines for item in sublist]
    [agents, tasks] = [int(numbers.pop(0)), int(numbers.pop(0))]
    profit_matrix = [[0 for i in range(tasks)] for j in range(agents)]
    cost_matrix = [[0 for i in range(tasks)] for j in range(agents)]

    for i in range(agents):
        for j in range(tasks):
            profit_matrix[i][j] = int(numbers.pop(0))

    for i in range(agents):
        for j in range(tasks):
            cost_matrix[i][j] = int(numbers.pop(0))

    capacity = [int(number) for number in numbers]

    return [agents, tasks, profit_matrix, cost_matrix, capacity]


def load_files(files):
    return dict((file_name, read_file(file_name)) for file_name in files)


def run():
    files = list_files(BASE_PATH)
    problem_inputs = load_files(files)
    solutions = [[*solve(key, *problem_inputs[key])] for key in files]
    write_solutions(solutions)


if __name__ == '__main__':
    run()
