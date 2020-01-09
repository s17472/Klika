import itertools as it
from math import factorial
import matplotlib.pyplot as plt
import numpy as np
import random as rm
from xlwings import xrange
import json
import collections
import sys
import timeit
from tqdm import tqdm

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
    def __init__(self, universe, sets, iterations, tabu_size, n_size):
        self.universe = universe
        self.sets = sets
        self.iterations = iterations
        self.tabu_size = tabu_size
        self.n_size = n_size

    @staticmethod
    def load_test_data(name):
        file = []
        with open(name, "r+") as text_file:
            file = text_file.read().splitlines()
        universe = [int(x) for x in file[0].split(',')]
        sets = json.loads(file[1], object_hook=json_as_python_set)
        iterations = int(file[2])
        tabu_size = int(file[3])
        n_size = int(file[4])
        return TestData(universe, sets, iterations, tabu_size, n_size)


def save_sets(S, name="Output.txt"):
    with open(name, "w") as text_file:
        serialized = json.dumps(S, cls=JSONSetEncoder)
        text_file.write(serialized)


def load_sets(name="Output.txt"):
    with open(name, "r+") as text_file:
        return json.loads(text_file.read(), object_hook=json_as_python_set)





def init_test_data_files():
    save_sets(generate_sets(U, 10), "10.txt")
    save_sets(generate_sets(U, 12), "12.txt")
    save_sets(generate_sets(U, 14), "14.txt")
    save_sets(generate_sets(U, 16), "16.txt")
    save_sets(generate_sets(U, 20), "20.txt")


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
    combos = rm.sample(xrange(1, int(combination_count + 1)), S_size)
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


def hill_full(s, it, n_size):
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

    return [best_score, get_nth_combination(s, best_combo)]
    # return [best_score]


def hill_random(s, it, n_size):
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

    return [best_score, get_nth_combination(s, best_combo)]
    # return [best_score]


def tabu(s, it, t_size, n_size):
    start = rm.randint(0, get_combination_count(len(s)) + 1)
    start_combo = get_nth_combination(s, start)
    best_score = goal(start_combo)
    best_combo = start
    checked = [start]

    for i in tqdm(range(it)):
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

    return [best_score, get_nth_combination(s, best_combo)]
    # return [best_score]


def benchmark(fun, name):
    start = timeit.default_timer()
    print(fun())
    took = (timeit.default_timer() - start) * 1000.0
    took = took / time_units["ms"]
    print("{:.5f}".format(took))


S2 = [{0, 3, 5}, {0, 1, 4}, {1, 4, 5}, {1, 2}, {0, 3}]
U = [1, 2, 3, 4, 5]
k = 1500
test2 = TestData.load_test_data('10.txt')
init_test_data_files()
xd = load_test_data_files()


test = TestData(U, xd['10'], 500, 600, 4)
benchmark(lambda: tabu(S2, k, 1500, 8), "tabu")
benchmark(lambda: tabu(S2, k, 1000, 8), "tabu")
benchmark(lambda: hill_full(S2, k, 8), "hill_full")
benchmark(lambda: hill_random(S2, k, 2), "hill_random")
