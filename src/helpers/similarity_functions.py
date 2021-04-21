import numpy as np


def cosine(active_user: np.array, user: np.array):
    users_intersection = [
        user[i] != 0 and active_user[i] != 0
        for i in range(len(user))
    ]

    u = active_user[users_intersection]
    v = user[users_intersection]

    if len(u) == 0 or len(v) == 0:
        return 0

    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
