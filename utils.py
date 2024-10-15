from itertools import product
import time
import mlrose_hiive as mlrose
import numpy as np
from joblib import Parallel, delayed

seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Call the decorated function
        end_time = time.time()  # Record end time
        execution_time = end_time - start_time  # Calculate execution time
        return result, execution_time  # Return result and execution time as a tuple

    return wrapper


def generate_parameter_list(parameter_dict):
    combinations = product(*parameter_dict.values())
    parameters_list = [
        dict(zip(parameter_dict.keys(), combination)) for combination in combinations
    ]

    return parameters_list


@timeit
def rhc_run(problem, max_attempt, max_iter, restart):
    results = []
    for seed in seeds:
        rhc = mlrose.random_hill_climb(
            problem,
            max_attempts=max_attempt,
            max_iters=max_iter,
            restarts=restart,
            init_state=None,
            curve=True,
            random_state=seed,
        )
        results.append(rhc)

    return results
    
@timeit
def sa_run(problem, decay, max_attempt, max_iter):
    results = []
    for seed in seeds:
        sa = mlrose.simulated_annealing(
            problem,
            schedule=decay,
            max_attempts=max_attempt,
            max_iters=max_iter,
            init_state=None,
            curve=True,
            random_state=seed,
        )
        results.append(sa)
    return results


@timeit
def ga_run(problem, pop_size, mutation_prob, max_attempt, max_iter):
    results = []
    for seed in seeds:
        ga = mlrose.genetic_alg(
            problem,
            pop_size=pop_size,
            mutation_prob=mutation_prob,
            max_attempts=max_attempt,
            max_iters=max_iter,
            curve=True,
            random_state=seed,
        )
        results.append(ga)

    return results


@timeit
def mimic_run(problem, pop_size, keep_pct, max_attempt, max_iter):
    results = []
    for seed in seeds:
        mimic = mlrose.mimic(
            problem,
            pop_size=pop_size,
            keep_pct=keep_pct,
            max_attempts=max_attempt,
            max_iters=max_iter,
            curve=True,
            random_state=seed,
        )
        results.append(mimic)

    return results


def fitness_mean_std(results):
    array = [x[1] for x in results]
    return np.mean(array), np.std(array)


def get_best_param(results, params):
    results_only = [x[0] for x in results]
    fitness_by_param = [fitness_mean_std(x) for x in results_only]
    best_param = np.argmax(fitness_by_param, axis=0)[0]

    return params[best_param]


def tune_params(grid, func, verbose=0):
    params = generate_parameter_list(grid)
    print("Number of params:", len(params))
    results = Parallel(n_jobs=-1, verbose=verbose)(
        delayed(func)(**params) for params in params
    )
    best_param = get_best_param(results, params)
    return best_param


def avg_with_impute(results, index):
    # Find the maximum length among all arrays
    if index == "fitness" or index == 0:
        list_of_arrays = [x[2][:, 0] for x in results]
    elif index == "feval" or index == 1:
        list_of_arrays = [x[2][:, 1] for x in results]

    max_length = max(arr.shape[0] for arr in list_of_arrays)

    # Pad each array individually to match the max length
    padded_arrays = []
    for arr in list_of_arrays:
        # Calculate the padding needed for the current array
        pad_length = max_length - arr.shape[0]
        # Pad the array and append it to the list of padded arrays
        if pad_length > 0:
            # We use np.pad with mode='edge' to extend the array with its last value
            padded_array = np.pad(arr, (0, pad_length), mode="edge")
        else:
            padded_array = arr
        padded_arrays.append(padded_array)

    # Stack the padded arrays and calculate the mean across the first axis
    stacked_arrays = np.vstack(padded_arrays)
    averages = np.mean(stacked_arrays, axis=0)
    std = np.std(stacked_arrays, axis=0)

    return averages, std


def avg_epoch(results):
    return np.mean([len(x[2]) for x in results])


def epoch_stats(results):
    _mean = np.mean([len(x[2]) for x in results])
    _max = np.max([len(x[2]) for x in results])
    _std = np.std([len(x[2]) for x in results])

    return _mean, _std, _max
