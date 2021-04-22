import numpy as np
import pandas as pd

from src.helpers.similarity_functions import cosine, mean_squared_distance, pearson, spearman
from src.modules.model import Model


def calculate_mean_error(ys, predicted_ys):
    return np.sum(np.abs(predicted_ys - ys)) / len(ys)


class NaiveNeighbourMethod(Model):
    def __init__(self, ratings_matrix: pd.DataFrame, similarity_method: str = 'cosine'):
        self.ratings_matrix = ratings_matrix
        self.similarity_method = similarity_method

        self.similarity_functions = {
            'cosine': cosine,
            'mean_squared_distance': mean_squared_distance,
            'pearson': pearson,
            'spearman': spearman
        }
        self.similarity_function = self.similarity_functions[
            self.similarity_method
        ]

    def _compute_weighted_average(self, num_neighbors: int, movie_id: int, similarities: list):
        nominator = 0
        dominator = 0

        for i in range(num_neighbors):
            similarity = similarities[i][0]
            user_id = similarities[i][1]

            nominator += similarity * self.ratings_matrix.loc[user_id, movie_id]
            dominator += np.abs(similarity)

        return nominator / dominator

    def train(self, ratings_matrix: pd.DataFrame):
        pass

    def compute_similarities(self, active_user_id: int):
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

        return similarities

    def predict(self, active_user_id: int, movies_ids: list, similarities=None, num_neighbors=10):
        if similarities is None:
            similarities = self.compute_similarities(active_user_id)

        return [
            self._compute_weighted_average(
                num_neighbors, movie_id, similarities
            )
            for movie_id in movies_ids
        ]
