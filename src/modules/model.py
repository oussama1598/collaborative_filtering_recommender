from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass
