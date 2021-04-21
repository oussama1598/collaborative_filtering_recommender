from abc import ABC, abstractmethod

import pandas as pd


class Model(ABC):
    @abstractmethod
    def train(self, ratings_matrix: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, active_user_id: int, movies_ids: list):
        pass
