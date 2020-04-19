from typing import List

import torch

from model.neuralCF import neural_helper
from model.neuralCF.neural_helper import TrainConfig
from tool.data_reader import Rating


class BaseModel(torch.nn.Module):
    def __init__(self, train_config: TrainConfig):
        super().__init__()
        self.current_epoch = 0
        self.train_loss = dict()
        self.train_config = train_config

    def fit(self, ratings: List[Rating]):
        neural_helper.train_neural(self, ratings)

    def predict(self, user_id: int, movie_id: int) -> float:
        return neural_helper.predict(self, user_id, movie_id)

    def predict_many(self, user_ids: List[int], movie_ids: List[int]) -> List[float]:
        return neural_helper.predict_many(self, user_ids, movie_ids)

    def get_device(self):
        return list(self.parameters())[0].device
