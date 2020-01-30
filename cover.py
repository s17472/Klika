import itertools as it
from math import factorial
import matplotlib.pyplot as plt
import numpy as np
import random as rm
import json
import collections
import sys
import timeit
from tqdm import tqdm
import pandas as pd


time_units = {'ms': 1, 's': 1000, 'm': 60 * 1000, 'h': 3600 * 1000}



class CodeTimer:
    def __init__(self, name=None, silent=False, unit='ms', logger_func=None):
        """Allows giving indented blocks their own name. Blank by default"""
        self.name = name
        self.silent = silent
        self.unit = unit
        self.logger_func = logger_func

    def __enter__(self):
        """Start measuring at the start of indent"""
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        """
            Stop measuring at the end of indent. This will run even
            if the indented lines raise an exception.
        """
        self.took = (timeit.default_timer() - self.start) * 1000.0
        self.took = self.took / time_units.get(self.unit, time_units['ms'])

        if not self.silent:
            log_str = '{}took: {:.5f} {}'.format(
                str(self.name + " ") if self.name else '',
                float(self.took),
                str(self.unit))

            if self.logger_func:
                self.logger_func(log_str)
            else:
                print(log_str)


# region Json
class JSONSetEncoder(json.JSONEncoder):
    """Use with json.dumps to allow Python sets to be encoded to JSON

    Example
    -------

    import json

    data = dict(aset=set([1,2,3]))

    encoded = json.dumps(data, cls=JSONSetEncoder)
    decoded = json.loads(encoded, object_hook=json_as_python_set)
    assert data == decoded     # Should assert successfully

    Any object that is matched by isinstance(obj, collections.Set) will
    be encoded, but the decoded value will always be a normal Python set.

    """

    def default(self, obj):
        if isinstance(obj, collections.Set):
            return dict(_set_object=list(obj))
        else:
            return json.JSONEncoder.default(self, obj)


def json_as_python_set(dct):
    """Decode json {'_set_object': [1,2,3]} to set([1,2,3])

    Example
    -------
    decoded = json.loads(encoded, object_hook=json_as_python_set)

    Also see :class:`JSONSetEncoder`

    """
    if '_set_object' in dct:
        return set(dct['_set_object'])
    return dct
# endregion


class TestData:
    def __init__(self, name, universe, sets, iterations, tabu_size, n_size):
        self.name = name
        self.universe = universe
        self.sets = sets
        self.iterations = iterations
        self.tabu_size = tabu_size
        self.n_size = n_size

    @staticmethod
    def load_test_data(name):
        file = []
        with open(name+".txt", "r+") as text_file:
            file = text_file.read().splitlines()
        universe = [int(x) for x in file[0].split(',')]
        sets = json.loads(file[1], object_hook=json_as_python_set)
        iterations = int(file[2])
        tabu_size = int(file[3])
        n_size = int(file[4])
        return TestData(name, universe, sets, iterations, tabu_size, n_size)

    def save_to_file(self):
        sets = json.dumps(self.sets, cls=JSONSetEncoder)
        with open(self.name + ".txt", "w") as text_file:
            text_file.write(','.join(map(str, self.universe))+"\n")
            text_file.write(sets+"\n")
            text_file.write(str(self.iterations)+"\n")
            text_file.write(str(self.tabu_size)+"\n")
            text_file.write(str(self.n_size)+"\n")


def save_sets(S, name="Output.txt"):
    with open(name, "w") as text_file:
        serialized = json.dumps(S, cls=JSONSetEncoder)
        text_file.write(serialized)


def load_sets(name="Output.txt"):
    with open(name, "r+") as text_file:
        return json.loads(text_file.read(), object_hook=json_as_python_set)


def init_test_data_files():
    with open("10.txt", "w") as f:
        f.write("""0,1,2,3,4,5
[{"_set_object": [4, 5]}, {"_set_object": [3, 4, 5]}, {"_set_object": [1, 2, 4]}, {"_set_object": [1, 2, 3]}, {"_set_object": [2]}, {"_set_object": [1, 4]}, {"_set_object": [4]}, {"_set_object": [2, 5]}, {"_set_object": [5]}, {"_set_object": [2, 3]}]
500
600
3""")
    with open("12.txt", "w") as f:
        f.write("""0,1,2,3,4,5
[{"_set_object": [4]}, {"_set_object": [5]}, {"_set_object": [4, 5]}, {"_set_object": [1, 4, 5]}, {"_set_object": [3, 4]}, {"_set_object": [1]}, {"_set_object": [3, 5]}, {"_set_object": [3]}, {"_set_object": [1, 2, 3]}, {"_set_object": [1, 4]}, {"_set_object": [2, 5]}, {"_set_object": [2]}]
500
600
3""")
    with open("14.txt", "w") as f:
        f.write("""0,1,2,3,4,5
[{"_set_object": [4]}, {"_set_object": [2, 5]}, {"_set_object": [3, 4]}, {"_set_object": [1, 2, 3]}, {"_set_object": [1, 2, 3, 4, 5]}, {"_set_object": [1, 2, 4, 5]}, {"_set_object": [2, 4]}, {"_set_object": [3, 5]}, {"_set_object": [2, 3, 4]}, {"_set_object": [1, 3]}, {"_set_object": [1, 3, 5]}, {"_set_object": [2, 3]}, {"_set_object": [1, 2]}, {"_set_object": [5]}]
500
600
3""")
    with open("16.txt", "w") as f:
        f.write("""0,1,2,3,4,5
[{"_set_object": [1, 4]}, {"_set_object": [2, 3, 5]}, {"_set_object": [3, 5]}, {"_set_object": [1]}, {"_set_object": [1, 3, 5]}, {"_set_object": [3, 4, 5]}, {"_set_object": [2]}, {"_set_object": [5]}, {"_set_object": [1, 3, 4, 5]}, {"_set_object": [1, 4, 5]}, {"_set_object": [2, 3, 4]}, {"_set_object": [1, 2, 4]}, {"_set_object": [1, 2, 3]}, {"_set_object": [3]}, {"_set_object": [2, 5]}, {"_set_object": [2, 3, 4, 5]}]
500
600
3""")
    with open("20.txt", "w") as f:
        f.write("""0,1,2,3,4,5
[{"_set_object": [2, 3]}, {"_set_object": [3, 4, 5]}, {"_set_object": [1, 5]}, {"_set_object": [3, 5]}, {"_set_object": [5]}, {"_set_object": [3, 4]}, {"_set_object": [1, 2, 3, 5]}, {"_set_object": [1, 2, 4, 5]}, {"_set_object": [2]}, {"_set_object": [2, 3, 4]}, {"_set_object": [1, 4, 5]}, {"_set_object": [1, 2, 3]}, {"_set_object": [2, 3, 4, 5]}, {"_set_object": [2, 5]}, {"_set_object": [1, 4]}, {"_set_object": [1, 2, 4]}, {"_set_object": [1, 2, 5]}, {"_set_object": [1]}, {"_set_object": [3]}, {"_set_object": [1, 2]}]
500
600
3""")


def load_test_data_files():
    return {'10': load_sets("10.txt"), '12': load_sets("12.txt"), '14': load_sets("14.txt"),
            '16': load_sets("16.txt"), '20': load_sets("20.txt")}


def goal(s):
    if not s: return 0
    union = set.union(*s)
    return len(union) - len(s) * 0.1


def get_combinations(s):
    combos = []
    for i in tqdm(range(1, len(s) + 1)):
        combos += list(it.combinations(s, i))
    return combos


def generate_sets(U, S_size):
    combination_count = get_combination_count(len(U))
    if combination_count > sys.maxsize:
        raise Exception(f"combination_count is over {sys.maxsize}")

    # print(combination_count)
    combos = rm.sample(range(1, int(combination_count + 1)), S_size)
    # print(combos)
    ret = [set(get_nth_combination(U, i)) for i in combos]
    return ret


def get_nth_combination(s, n):
    combination_count = get_combination_count(len(s))
    if not (0 < n <= combination_count):
        n = int(n % combination_count)
        # print(f"n is <= 0 or exceeds possible combinations count, n = {n}")

    bin_str = bin(n)[:1:-1]
    combo = []
    for i in range(len(bin_str)):
        if int(bin_str[i]) == 1:
            combo.append(s[i])

    return combo


def get_neighbourhood_full(s, start):
    return {start - 2: goal(get_nth_combination(s, start - 2)),
            start - 1: goal(get_nth_combination(s, start - 1)),
            start + 1: goal(get_nth_combination(s, start + 1)),
            start + 2: goal(get_nth_combination(s, start + 2))}


def get_neighbourhood_close(s, start):
    return {start - 1: goal(get_nth_combination(s, start - 1)),
            start + 1: goal(get_nth_combination(s, start + 1))}


def get_neighbourhood(s, start, n):
    ret = {}
    for i in [x for x in range(start - n, start + n + 1) if x != start]:
        ret[i] = goal(get_nth_combination(s, i))
    return ret


def get_combination_count(s_len):
    combination_count = 0
    for i in range(s_len):
        combination_count += factorial(s_len) / (factorial(i) * factorial(s_len - i))
    return combination_count


def brute(s):
    combos = get_combinations(s)
    best_score = 0
    best_combo = combos[0]

    for i in (range(len(combos))):
        score = goal(combos[i])
        if score > best_score:
            best_combo = combos[i]
            best_score = score

    return [best_score, get_nth_combination(s, best_index)]


def hill_full(s, it, tabu_size, n_size):
    start = rm.randint(0, get_combination_count(len(s)) + 1)
    start_combo = get_nth_combination(s, start)
    best_score = goal(start_combo)
    best_combo = start
    # print(f"{best_score}: {get_nth_combination(s, best_combo)} : {best_combo}")

    for i in (range(it)):
        n = get_neighbourhood(s, start, n_size)
        best_n_index = max(n, key=n.get)
        best_n_score = n[best_n_index]
        if best_score > best_n_score:
            break
        best_score = best_n_score
        best_combo = best_n_index
        start = best_n_index
        # print(f"{best_score}: {get_nth_combination(s, best_combo)} : {best_combo}")

    # return [best_score, get_nth_combination(s, best_combo)]
    return best_score


def hill_random(s, it, tabu_size, n_size):
    n_size = n_size // 3
    start = rm.randint(0, get_combination_count(len(s)) + 1)
    start_combo = get_nth_combination(s, start)
    best_score = goal(start_combo)
    best_combo = start
    # print(f"{best_score}: {get_nth_combination(s, best_combo)} : {best_combo}")

    for i in (range(it)):
        n = get_neighbourhood(s, start, n_size)
        best_n_index = rm.choice(list(n.keys()))
        best_n_score = n[best_n_index]
        if best_score > best_n_score:
            break
        best_score = best_n_score
        best_combo = best_n_index
        start = best_n_index
        # print(f"{best_score}: {get_nth_combination(s, best_combo)} : {best_combo}")

    # return [best_score, get_nth_combination(s, best_combo)]
    return best_score


def tabu(s, it, t_size, n_size):
    start = rm.randint(0, get_combination_count(len(s)) + 1)
    start_combo = get_nth_combination(s, start)
    best_score = goal(start_combo)
    best_combo = start
    checked = [start]

    for i in (range(it)):
        n = get_neighbourhood(s, start, n_size)
        n_tabu = {key: value for (key, value) in n.items() if key not in checked}
        if not n_tabu:
            break
        best_n_index = max(n_tabu, key=n.get)
        checked.append(best_n_index)
        best_n_score = n_tabu[best_n_index]
        # print(f"C: {best_n_index}: {best_n_score}: {get_nth_combination(s, best_n_index)}")
        start = best_n_index
        if best_score <= best_n_score:
            best_score = best_n_score
            best_combo = best_n_index
        if len(checked) >= t_size:
            break
        # print(f"B: {best_index}: {best_score}: {get_nth_combination(s, best_index)}")

    # return [best_score, get_nth_combination(s, best_combo)]
    return best_score


def benchmark(fun, test_data, iterations, name):
    score_avg = 0
    start = timeit.default_timer()
    for i in range(iterations):
        score_avg += fun(test_data.sets, test_data.iterations, test_data.tabu_size, test_data.n_size)
    took = (timeit.default_timer() - start) * 1000.0
    took = took / time_units["ms"] / iterations
    print("{} {} {:.5f} {:.2f}".format(name, test_data.name, took/iterations, score_avg/iterations))


def benchmark_(funs, data, iterations):
    print("Nazwa rozmiar czas wynik wynik/czas")
    res = []
    for fun in funs:
        for test_data in data:
            score_avg = 0
            start = timeit.default_timer()
            for i in range(iterations):
                score_avg += fun(test_data.sets, test_data.iterations, test_data.tabu_size, test_data.n_size)
            took = (timeit.default_timer() - start) * 1000.0
            took = took / time_units["ms"] / iterations
            print("{} {} {:.4f} {:.2f} {:.2f}".format(fun.__name__, test_data.name, round(took/iterations, 4), round(score_avg/iterations, 2), (score_avg/iterations)/(took/iterations)))
            res.append([fun.__name__, test_data.name, round(took/iterations, 4), round(score_avg/iterations, 2)])
    return res


def plot_(xd):
    pf = pd.DataFrame(xd, columns=['name', 'size', 'time', 'score'])
    x = pf.name.unique()
    # sizes = pf.size.unique()
    # y = []
    # for s in sizes:
    #     plt.plot(x, pf.loc[pf['size'] == '10'].time.values)

    y10 = pf.loc[pf['size'] == '10'].time.values
    y12 = pf.loc[pf['size'] == '12'].time.values
    y14 = pf.loc[pf['size'] == '14'].time.values
    y16 = pf.loc[pf['size'] == '16'].time.values
    # plt.xticks(np.array(range(3)), x)
    plt.plot(x, y10)
    plt.plot(x, y12)
    plt.plot(x, y14)
    plt.plot(x, y16)
    plt.show()


def plot_input(input):
    plt.title("Input data")
    plt.yticks(np.arange(len(S2)))
    plt.ylabel("Set ID")
    plt.xlabel("Universe")
    for i in range(len(input)):
        plt.scatter(list(input[i]), list(it.repeat(i, len(input[i]))), s=1000)
    plt.show()


def generate_random_solution(sets):
    return get_nth_combination(sets, rm.randint(0, get_combination_count(len(sets)) + 1))


def plot_result(input, result):
    plt.title("Result")
    plt.yticks(np.arange(len(S2)))
    plt.ylabel("Set ID")
    plt.xlabel("Universe")
    for i in range(len(input)):
        if input[i] in result:
            plt.scatter(list(input[i]), list(it.repeat(i, len(input[i]))), s=1000, c='green')
        else:
            plt.scatter(list(input[i]), list(it.repeat(i, len(input[i]))), s=1000, c='grey')
    plt.show()


def initialize_population(sets, size):
    population = []
    for i in range(size):
        population.append(generate_random_solution(sets))
    return population


def fitness(solution):
    return 1000.0 / (1.0 + goal(solution))


def selection(population):
    first = rm.choice(population)
    second = rm.choice(population)
    return first if fitness(first) > fitness(second) else second


def crossover(parent_a, parent_b):
    shorter = min(len(parent_a), len(parent_b))
    cross_point = round(rm.uniform(0, shorter))
    child_a = [parent_a[:cross_point:], parent_b[cross_point::]]
    child_b = [parent_b[:cross_point:], parent_a[cross_point::]]

    return child_a, child_b


def mutation(sets, specimen):
    avalable = [x for x in sets if x not in specimen]
    if avalable:
        return specimen

    new = rm.choice(avalable)
    old_index = rm.randint(0, len(specimen))
    specimen.pop(old_index)
    specimen.append(new)
    return specimen


def termination(population, iteration):
    return len(population) != iteration


def generic(sets, initial_population, fitness_func, selection_func, crossover_func, mutation_func, termination_func, crossover_probability = 0.9, mutation_probability = 0.1):
    population = initial_population

    i = 0
    while termination_func(population, i):
        fits = []
        parents = []
        children = []
        for specimen in population:
            fits.append(fitness_func(specimen))

        for s in range(len(initial_population)):
            parents.append(selection_func(population))

        it = 0
        while it < len(initial_population):
            cross = rm.uniform(0, 1) < crossover_probability
            if cross:
                a, b = crossover_func(parents[it], parents[it+1])
                children.append(a)
                children.append(b)
            else:
                children.append(parents[i])
                children.append(parents[i+1])
            it += 2

        it = 0
        while it < len(initial_population):
            mutate = rm.uniform(0, 1) < mutation_probability
            if mutate:
                children[it] = mutation_func(sets, children[i])
            it += 2

        population = children
        i += 1

    return max(population, key=fitness_func)



S2 = [{0, 3, 5}, {0, 1, 4}, {1, 4, 5}, {1, 2}, {0, 3}]
U = [-1, 0, 1, 2, 3, 4, 5]
k = 1500
test10 = TestData.load_test_data('10')
# test12 = TestData.load_test_data('12')
# test14 = TestData.load_test_data('14')
# test16 = TestData.load_test_data('16')
# test20 = TestData.load_test_data('20')

# benchmark(tabu, test10, 25, "Tabu")
# benchmark(tabu, test12, 25, "Tabu")
# a = benchmark_([tabu, hill_full, hill_random], [test10, test12, test14, test16], 25)
# plot_(a)

# benchmark(tabu, test16, 25, "Tabu")
# benchmark(tabu, test20, 25, "Tabu")
# benchmark(lambda: tabu(S2, k, 1000, 8), "tabu")
# benchmark(lambda: hill_full(S2, k, 8), "hill_full")
# benchmark(lambda: hill_random(S2, k, 2), "hill_random")


R2 = [{0, 3, 5}, {0, 1, 4}, {1, 2}]
# r = tabu(S2, 1000, 1000, 4)

init_pop = initialize_population(test10.sets, 10)

generic(S2, init_pop, fitness, selection, crossover, mutation, termination,
        crossover_probability=0.9,
        mutation_probability=0.1)

plot_input(S2)
plot_result(S2, R2)
