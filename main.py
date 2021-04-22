import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.modules.naive_neighbour_method import NaiveNeighbourMethod, calculate_mean_error

df_ratings = pd.read_csv(
    './data/ratings.csv',
    usecols=['userId', 'movieId', 'rating'])
# filter data
df_movies_cnt = pd.DataFrame(
    df_ratings.groupby('movieId').size(),
    columns=['count'])
popular_movies = list(set(df_movies_cnt.query('count >= 300').index))
movies_filter = df_ratings.movieId.isin(popular_movies).values

df_users_cnt = pd.DataFrame(
    df_ratings.groupby('userId').size(),
    columns=['count'])
active_users = list(set(df_users_cnt.query('count >= 200').index))
users_filter = df_ratings.userId.isin(active_users).values

df_ratings_filtered = df_ratings[movies_filter & users_filter]

ratings_matrix = df_ratings_filtered.pivot(
    index='userId', columns='movieId', values='rating'
).fillna(0)

# ratings = pd.read_csv(
#     './data/ratings.csv',
#     usecols=['userId', 'movieId', 'rating']
# )
# ratings_matrix = ratings.pivot(
#     index='userId', columns='movieId', values='rating'
# ).fillna(0)

predictor = NaiveNeighbourMethod(
    similarity_method='cosine',
    ratings_matrix=ratings_matrix
)

user_id = 1
neighbors = range(10, 130, 10)
losses = []
similarities = predictor.compute_similarities(user_id)

for i in neighbors:
    user_vector = np.array(ratings_matrix.loc[user_id])
    predictions = predictor.predict(
        user_id,
        ratings_matrix.columns,
        similarities=similarities,
        num_neighbors=i
    )

    losses.append(
        calculate_mean_error(user_vector, predictions)
    )

plt.plot(neighbors, losses)
plt.show()
