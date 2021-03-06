from typing import List

import numpy as np

from tool.data_reader import Rating, generate_rating_matrix


class UserCF(object):
    def __init__(self, knn_k: int):
        self.M = None
        self.KNN_K = knn_k

    def fit(self, ratings_train: List[Rating]):
        self.M = generate_rating_matrix(ratings_train)

    def calculate_avg_rating_for_user(self, i):
        return self.M[i, :].sum() / (self.M[i, :] != 0).sum()

    def calculate_user_sim(self, i, j):
        movies_i_like = set(np.where(self.M[i, :] != 0)[0])
        movies_j_like = set(np.where(self.M[j, :] != 0)[0])
        both = sorted(movies_i_like & movies_j_like)
        if len(both) == 0:
            return 0
        else:
            rating_by_i = self.M[i, list(both)]
            rating_by_j = self.M[j, list(both)]
            return np.corrcoef(rating_by_i, rating_by_j)[0, 1]

    def predict(self, user_id: int, movie_id: int) -> float:
        neighbors = []

        for i in range(self.M.shape[0]):
            if self.M[i, movie_id] != 0:
                sim = self.calculate_user_sim(i, user_id)
                neighbors.append((i, sim))

        neighbors = [n for n in neighbors if n[1] > 0]
        neighbors = sorted(neighbors, key=lambda n: -n[1])[:self.KNN_K]
        if len(neighbors) == 0:
            return 0

        total_sim = 0
        rating_predict = 0
        for i, sim in neighbors:
            total_sim += sim
            rating_predict += (self.M[i, movie_id] - self.calculate_avg_rating_for_user(i)) * sim
        rating_predict /= total_sim
        rating_predict += self.calculate_avg_rating_for_user(user_id)

        return rating_predict
