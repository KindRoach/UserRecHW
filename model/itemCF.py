import numpy as np


class ItemCF(object):
    def __init__(self, ratings_matrix: np.ndarray, knn_k: int):
        self.M = ratings_matrix.T
        self.KNN_K = knn_k

    def train(self):
        pass

    def calculate_movie_sim(self, i: int, j: int):
        users_like_i = set(np.where(self.M[i, :] != 0)[0])
        users_like_j = set(np.where(self.M[j, :] != 0)[0])
        both = users_like_i & users_like_j
        if len(users_like_i) == 0 or len(users_like_j) == 0:
            return 0
        else:
            return len(both) / np.math.sqrt(len(users_like_i) * len(users_like_j))

    def predict(self, user_id: int, movie_id: int) -> float:
        neighbors = []

        for i in range(self.M.shape[0]):
            if self.M[i, user_id - 1] != 0:
                sim = self.calculate_movie_sim(i, movie_id - 1)
                neighbors.append((i, sim))

        total_sim = 0
        rating_predict = 0
        neighbors = sorted(neighbors, key=lambda n: -n[1])[:self.KNN_K]
        for i, sim in neighbors:
            total_sim += sim
            rating_predict += self.M[i, user_id - 1] * sim
        rating_predict /= total_sim

        return rating_predict
