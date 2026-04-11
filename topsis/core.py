import pandas as pd
import numpy as np


def validate_inputs(data, weights, impacts):
    if data.shape[1] < 3:
        raise ValueError("Input file must contain at least 3 columns")

    matrix = data.iloc[:, 1:]

    if not all(np.issubdtype(dtype, np.number) for dtype in matrix.dtypes):
        raise ValueError("All columns except the first must contain numeric values only")

    if len(weights) != matrix.shape[1] or len(impacts) != matrix.shape[1]:
        raise ValueError(
            f"Number of weights ({len(weights)}) and impacts ({len(impacts)}) "
            f"must be equal to the number of criteria columns ({matrix.shape[1]})"
        )

    if not all(i in ['+', '-'] for i in impacts):
        raise ValueError("Impacts must be '+' or '-' only")

    if not all(w > 0 for w in weights):
        raise ValueError("All weights must be positive numbers")


def normalize_matrix(matrix):
    return matrix / np.sqrt((matrix ** 2).sum())


def apply_weights(normalized_matrix, weights):
    return normalized_matrix * weights


def get_ideal_solutions(weighted_matrix, impacts):
    ideal_best = []
    ideal_worst = []

    for i, impact in enumerate(impacts):
        col = weighted_matrix.iloc[:, i]
        if impact == '+':
            ideal_best.append(col.max())
            ideal_worst.append(col.min())
        else:
            ideal_best.append(col.min())
            ideal_worst.append(col.max())

    return np.array(ideal_best), np.array(ideal_worst)


def calculate_scores(weighted_matrix, ideal_best, ideal_worst):
    dist_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))
    scores = dist_worst / (dist_best + dist_worst)
    return scores


def topsis(data, weights, impacts):
    validate_inputs(data, weights, impacts)

    matrix = data.iloc[:, 1:]

    norm_matrix = normalize_matrix(matrix)
    weighted_matrix = apply_weights(norm_matrix, weights)
    ideal_best, ideal_worst = get_ideal_solutions(weighted_matrix, impacts)
    scores = calculate_scores(weighted_matrix, ideal_best, ideal_worst)

    result = data.copy()
    result['Topsis Score'] = scores
    result['Rank'] = scores.rank(ascending=False, method='max').astype(int)

    return result
