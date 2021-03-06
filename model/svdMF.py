from typing import List

import numpy as np

from tool.data_reader import Rating, generate_rating_matrix


class SvdMF(object):
    def __init__(self, pac_p: float):
        self.M = None
        self.PAC_P = pac_p

        self.U = None
        self.S = None
        self.V = None
        self.PAC_K = 0

    def fit(self, ratings_train: List[Rating]):
        self.M = generate_rating_matrix(ratings_train)
        self.U, self.S, self.V = np.linalg.svd(self.M, full_matrices=False)
        total_pac = self.S.sum()
        current_pac = 0
        for i in range(self.S.shape[0]):
            current_pac += self.S[i]
            if current_pac / total_pac > self.PAC_P:
                self.PAC_K = i
                break

        self.U = self.U[:, :self.PAC_K]
        self.S = np.diag(self.S[:self.PAC_K])
        self.V = self.V[:self.PAC_K, :]

    def calculate_avg_rating_for_user(self, i):
        return self.M[i, :].sum() / (self.M[i, :] != 0).sum()

    def predict(self, user_id: int, movie_id: int) -> float:
        avg = self.calculate_avg_rating_for_user(user_id)
        dot = self.U[user_id, :].dot(self.S).dot(self.V[:, movie_id])
        return avg + dot
