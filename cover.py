import itertools as it
from math import factorial
import matplotlib.pyplot as plt
import numpy as np
import random as rm
from xlwings import xrange
import json
import collections
import pickle


#region Json
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
#endregion


seed = 3555


def save_sets(S):
    with open("Output.txt", "w") as text_file:
        serialized = json.dumps(S, cls=JSONSetEncoder)
        text_file.write(serialized)


def load_sets():
    with open("Output.txt", "r+") as text_file:
        return json.loads(text_file.read(), object_hook=json_as_python_set)


def goal(s):
    if not s: return 0
    union = set.union(*s)
    return len(union) - len(s) * 0.1


def get_combinations(s):
    combos = []
    for i in range(1, len(s) + 1):
        combos += list(it.combinations(s, i))
    return combos


def generate_sets(U, S_size):
    rm.seed(7)
    combination_count = get_combination_count(len(U))
    print(combination_count)
    combos = [rm.randrange(0, combination_count+1) for _ in xrange(S_size)]
    print(combos)
    return [set(get_nth_combination(U, i)) for i in combos]


def brute(s):
    combos = get_combinations(s)
    best_score = 0
    best_combo = combos[0]

    for i in range(len(combos)):
        score = goal(combos[i])
        if score > best_score:
            best_combo = combos[i]
            best_score = score

    return best_combo


def get_nth_combination(s, n):
    combination_count = get_combination_count(len(s))
    if not (0 < n <= combination_count):
        n = int(n % combination_count)
        print(f"n is <= 0 or exceeds possible combinations count, n = {n}")

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
    for i in [x for x in range(start-n, start+n+1) if x != start]:
        ret[i] = goal(get_nth_combination(s, i))
    return ret


def get_combination_count(s_len):
    combination_count = 0
    for i in range(s_len):
        combination_count += factorial(s_len) / (factorial(i) * factorial(s_len - i))
    return combination_count


def hill_full(s, x):
    rm.seed(seed)
    start = rm.randint(0, get_combination_count(len(s))+1)
    start_combo = get_nth_combination(s, start)
    best_score = goal(start_combo)
    best_combo = start
    print(f"{best_score}: {get_nth_combination(s, best_combo)} : {best_combo}")

    for i in range(x):
        n = get_neighbourhood(s, start, 2)
        best_n_index = max(n, key=n.get)
        best_n_score = n[best_n_index]
        if best_score > best_n_score:
            break
        best_score = best_n_score
        best_combo = best_n_index
        start = best_n_index
        print(f"{best_score}: {get_nth_combination(s, best_combo)} : {best_combo}")

    return [get_nth_combination(s, best_combo), best_score]


def hill_random(s, x):
    rm.seed(seed)
    start = rm.randint(0, get_combination_count(len(s))+1)
    start_combo = get_nth_combination(s, start)
    best_score = goal(start_combo)
    best_combo = start

    for i in range(x):
        n = get_neighbourhood(s, start, 1)
        best_n_index = rm.choice(list(n.keys()))
        best_n_score = n[best_n_index]
        if best_score > best_n_score:
            break
        best_score = best_n_score
        best_combo = best_n_index
        start = best_n_index

    return [get_nth_combination(s, best_combo), best_score]


def tabu(s, it, t_MAX):
    rm.seed(seed)
    start = rm.randint(0, get_combination_count(len(s)) + 1)
    start_combo = get_nth_combination(s, start)
    best_score = goal(start_combo)
    checked = [start]

    for i in range(it):
        n = get_neighbourhood(s, start, 2)
        n_tabu = {key: value for (key, value) in n.items() if key not in checked}
        if not n_tabu:
            break
        best_n_index = max(n_tabu, key=n.get)
        checked.append(best_n_index)
        best_n_score = n_tabu[best_n_index]
        print(f"C: {best_n_index}: {best_n_score}: {get_nth_combination(s, best_n_index)}")
        start = best_n_index
        if best_score <= best_n_score:
            best_score = best_n_score
            best_index = best_n_index
        if len(checked) >= t_MAX:
            break
        print(f"B: {best_index}: {best_score}: {get_nth_combination(s, best_index)}")

    return [get_nth_combination(s, best_index), best_score]




S = [{2, 3}, {5}, {3, 4}, {4, 5}, {1, 2, 3}]
U = [1, 2, 3, 4, 5]
U2 = xrange(6)
# print(hill_full(S, 10))
# print(hill_random(S, 10))
# S2 = generate_sets(U2, 10)
# S3 = [{0, 3, 5}, {0, 1, 4}, {1, 4, 5}, {1, 2}, {0, 3}, {2, 3}, {1, 2, 3, 5}, {0, 1, 2}, {0, 1, 2, 3, 4, 5}, {5, 6}]
print(tabu(S, 10, 5))




# print(load_sets())
# S = [{2, 3}, {5}, {3, 4}, {4, 5}, {1, 2, 3}, {1, 2, 3, 4}, {55}]
# save_sets(S)
# print(load_sets())
# c = get_combinations(S)
# print(hill_full(S, 10))


