import numpy as np
import pandas as pd

from src.helpers.similarity_functions import cosine
from src.modules.model import Model


def calculate_loss(ys, predicted_ys):
    return np.sum([
        np.abs(predicted_ys[i] - ys[i])
        for i in range(len(ys))
    ]) / len(ys)


class NaiveNeighbourMethod(Model):
    def __init__(self, similarity_method: str = 'cosine', num_neighbors=10):
        self.ratings_matrix = None
        self.similarity_method = similarity_method
        self.num_neighbors = num_neighbors

        self.similarity_functions = {
            'cosine': cosine
        }
        self.similarity_function = self.similarity_functions[
            self.similarity_method
        ]

    def _compute_weighted_average(self, movie_id: int, similarities: list, previous_ratings_mean: float):
        nominator = 0
        dominator = 0

        for i in range(self.num_neighbors):
            similarity = similarities[i][0]
            user_id = similarities[i][1]

            user_rating_mean = np.mean(
                np.array(self.ratings_matrix.loc[user_id])
            )

            nominator += similarity * (self.ratings_matrix.loc[user_id, movie_id] - user_rating_mean)
            dominator += np.abs(similarity)

        return previous_ratings_mean + (nominator / dominator)

    def train(self, ratings_matrix: pd.DataFrame):
        self.ratings_matrix = ratings_matrix

    def predict(self, active_user_id: int, movies_ids: list):
        active_user_vector = np.array(self.ratings_matrix.loc[active_user_id])
        similarities = []

        for user_id in self.ratings_matrix.index:
            if user_id == active_user_id:
                continue

            user_vector = np.array(self.ratings_matrix.loc[user_id])

            similarities.append(
                (self.similarity_function(active_user_vector, user_vector), user_id)
            )

        similarities.sort(key=lambda x: x[0], reverse=True)

        return [
            self._compute_weighted_average(
                movie_id, similarities, np.mean(active_user_vector)
            )
            for movie_id in movies_ids
        ]
