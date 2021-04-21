import pandas as pd

from src.modules.naive_neighbour_method import NaiveNeighbourMethod

ratings = pd.read_csv(
    './data/ratings.csv',
    usecols=['userId', 'movieId', 'rating']
)
ratings_matrix = ratings.pivot(
    index='userId', columns='movieId', values='rating'
).fillna(0)

predictor = NaiveNeighbourMethod()
predictor.train(ratings_matrix)

predictor.predict(1, ratings_matrix.columns)
