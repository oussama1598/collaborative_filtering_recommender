import numpy as np


def get_intersections(u: np.array, v: np.array):
    users_intersection = [
        u[i] != 0 and v[i] != 0
        for i in range(len(u))
    ]

    return u[users_intersection], v[users_intersection]


def cosine(active_user: np.array, user: np.array):
    u, v = get_intersections(active_user, user)

    if len(u) == 0 or len(v) == 0:
        return 0

    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def mean_squared_distance(active_user: np.array, user: np.array):
    u, v = get_intersections(active_user, user)

    if len(u) == 0 or len(v) == 0:
        return 0

    return np.sum(np.power(u - v, 2)) / len(u)


def pearson(active_user: np.array, user: np.array):
    u, v = get_intersections(active_user, user)

    if len(u) == 0 or len(v) == 0:
        return 0

    a, b = u - np.mean(u), v - np.mean(v)

    nominator = np.sum(a * b)
    dominator = (np.sqrt(np.sum(np.power(a, 2))) * np.sqrt(np.sum(np.power(b, 2))))

    if dominator == 0:
        return 0

    return nominator / dominator


def spearman(active_user: np.array, user: np.array):
    u, v = get_intersections(active_user, user)

    if len(u) == 0 or len(v) == 0:
        return 0

    u, v = u.argsort() + 1, v.argsort() + 1
    a, b = u - np.mean(u), v - np.mean(v)

    nominator = np.sum(a * b)
    dominator = (np.sqrt(np.sum(np.power(a, 2))) * np.sqrt(np.sum(np.power(b, 2))))

    if dominator == 0:
        return 0

    return nominator / dominator
