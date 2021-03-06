import time
from dataclasses import dataclass
from typing import List

import torch
from torch.optim import lr_scheduler
from torch.utils import data

from tool.data_reader import Rating
from tool.log_helper import logger
from tool.path_helper import ROOT_DIR


@dataclass
class TrainConfig(object):
    num_epochs: int
    batch_size: int
    learning_rate: float
    l2_regularization: float
    use_cuda: bool


class BaseModel(torch.nn.Module):
    def __init__(self, train_config: TrainConfig):
        super().__init__()
        self.current_epoch = 0
        self.train_loss = dict()
        self.train_config = train_config

    def fit(self, ratings: List[Rating]):
        train_neural(self, ratings)

    def predict(self, user_id: int, movie_id: int) -> float:
        return predict(self, user_id, movie_id)

    def predict_many(self, user_ids: List[int], movie_ids: List[int]) -> List[float]:
        return predict_many(self, user_ids, movie_ids)

    def get_device(self):
        return list(self.parameters())[0].device


def train_neural(model: BaseModel, ratings: List[Rating]):
    logger.info("%s Training..." % model.__class__.__name__)
    train_time = time.localtime()

    config: TrainConfig = model.train_config

    if config.use_cuda:
        model.cuda()

    model.train()
    data_iter = generate_tensor_data(ratings, config.batch_size, model.get_device())
    opt = torch.optim.Adam(model.parameters(),
                           lr=config.learning_rate,
                           weight_decay=config.l2_regularization)
    lr_s = lr_scheduler.ExponentialLR(opt, gamma=0.9)
    loss = torch.nn.BCELoss()

    last_progress = 0.
    while model.current_epoch < config.num_epochs:
        for batch_id, iter_i in enumerate(data_iter):
            user, movie, rating = iter_i

            # train one step
            opt.zero_grad()
            li = loss(model(user, movie).view(-1), rating)
            li.backward()
            opt.step()

            # log progress
            current_batches = model.current_epoch * len(data_iter.dataset) + (batch_id + 1.0) * config.batch_size
            total_batches = config.num_epochs * len(data_iter.dataset)
            progress = current_batches / total_batches
            if progress - last_progress > 0.001:
                logger.info("epoch %d, batch %d, loss: %f (%.1f%%)" %
                            (model.current_epoch, batch_id, li.item(), 100.0 * progress))
                last_progress = progress

        # complete one epoch
        with torch.no_grad():
            all_user, all_movie, all_rating = data_iter.dataset.tensors
            total_loss = loss(model(all_user, all_movie).view(-1), all_rating)
            model.train_loss[model.current_epoch] = total_loss.item()
            logger.info("Epoch %d complete. Total loss=%f" % (model.current_epoch, total_loss.item()))

        lr_s.step(model.current_epoch)
        model.current_epoch += 1
        save_model(model, train_time)

    logger.info("%s Trained." % model.__class__.__name__)


def generate_tensor_data(ratings: List[Rating], batch_size: int, device: torch.device):
    user_tensor = torch.LongTensor([r.user_id for r in ratings]).to(device)
    movie_tensor = torch.LongTensor([r.movie_id for r in ratings]).to(device)
    rating_tensor = torch.FloatTensor([r.rating for r in ratings]).to(device)

    dataset = data.TensorDataset(user_tensor, movie_tensor, rating_tensor)
    data_iter = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_iter


def save_model(model: BaseModel, train_time: time.struct_time):
    config: TrainConfig = model.train_config
    path = "model/neuralCF/checkpoints/%s_%s_%d_%g_%g.pt" % (
        model.__class__.__name__, time.strftime("%Y%m%d%H%M%S", train_time)
        , config.batch_size, config.learning_rate, config.l2_regularization
    )
    path = ROOT_DIR.joinpath(path)
    torch.save(model, path)


def load_model(path: str):
    path = ROOT_DIR.joinpath(path)
    # load model to cpu as default.
    model = torch.load(path, map_location=torch.device('cpu'))
    return model


def predict(model, user_id: int, movie_id: int) -> int:
    device = model.get_device()
    user_id = torch.LongTensor([user_id]).to(device)
    movie_id = torch.LongTensor([movie_id]).to(device)
    predict = model(user_id, movie_id)[0]
    return predict


def predict_many(model, user_ids: List[int], movie_ids: List[int]) -> List[float]:
    device = model.get_device()
    user_ids = torch.LongTensor(user_ids).to(device)
    movie_ids = torch.LongTensor(movie_ids).to(device)
    predict = model(user_ids, movie_ids)
    return predict.tolist()
