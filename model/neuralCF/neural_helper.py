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


def generate_tensor_data(ratings: List[Rating], batch_size: int, use_cuda: bool):
    user_tensor = torch.LongTensor([r.user_id for r in ratings])
    movie_tensor = torch.LongTensor([r.movie_id for r in ratings])
    rating_tensor = torch.FloatTensor([r.rating for r in ratings])

    if use_cuda:
        user_tensor.cuda()
        movie_tensor.cuda()
        rating_tensor.cuda()

    dataset = data.TensorDataset(user_tensor, movie_tensor, rating_tensor)
    data_iter = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_iter


def train_neural(model: torch.nn.Module, ratings: List[Rating]):
    model.train()
    config: TrainConfig = model.train_config
    data_iter = generate_tensor_data(ratings, config.batch_size, config.use_cuda)
    opt = torch.optim.Adam(model.parameters(),
                           lr=config.learning_rate,
                           weight_decay=config.l2_regularization)
    lr_s = lr_scheduler.ExponentialLR(opt, gamma=0.9)

    loss = torch.nn.BCELoss()
    while model.current_epoch < config.num_epochs:
        for batch_id, iter_i in enumerate(data_iter):
            user, movie, rating = iter_i

            # train one step
            li = loss(model(user, movie).view(-1), rating)
            model.current_loss = li.item()
            opt.zero_grad()
            li.backward()
            opt.step()

            # log progress
            current_batches = model.current_epoch * len(data_iter.dataset) + (batch_id + 1.0) * config.batch_size
            total_batches = config.num_epochs * len(data_iter.dataset)
            progress = current_batches / total_batches
            logger.info("epoch %d, batch %d, loss: %f (%.1f%%)" %
                        (model.current_epoch, batch_id, model.current_loss, 100.0 * progress))

        # complete one epoch
        lr_s.step()
        model.current_epoch += 1
        save_model(model)


def save_model(model: torch.nn.Module):
    config: TrainConfig = model.train_config
    path = "model/neuralCF/checkpoints/%s_%d_%g_%g.pt" % (
        model.__class__.__name__, config.batch_size,
        config.learning_rate, config.l2_regularization
    )
    path = ROOT_DIR.joinpath(path)
    torch.save(model, path)


def load_model(path: str):
    path = ROOT_DIR.joinpath(path)
    model = torch.load(path)
    if model.train_config.use_cuda:
        model.cuda()
    return model
