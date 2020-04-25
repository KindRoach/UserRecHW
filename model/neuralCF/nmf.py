from typing import List

import torch

from model.neuralCF.neural_helper import TrainConfig, BaseModel


def get_default_config() -> TrainConfig:
    return TrainConfig(num_epochs=100,
                       batch_size=1024,
                       learning_rate=0.01,
                       l2_regularization=0.0,
                       use_cuda=False)


class NeuralMF(BaseModel):
    def __init__(self, num_users: int, num_items: int, latent_dim: int = 8,
                 hidden_layer_dims: List[int] = [64, 32, 16],
                 train_config: TrainConfig = None):

        if train_config is None:
            train_config = get_default_config()
        super(NeuralMF, self).__init__(train_config)

        self.num_users = num_users + 1
        self.num_items = num_items + 1
        self.latent_dim = latent_dim
        self.embedding_user_GMF = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item_GMF = torch.nn.Embedding(self.num_items, self.latent_dim)
        self.embedding_user_MLP = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item_MLP = torch.nn.Embedding(self.num_items, self.latent_dim)

        hidden_layer_dims.insert(0, latent_dim * 2)
        hidden_layer_dims.append(latent_dim)
        self.mlp_layer_dims = hidden_layer_dims
        self.mlp_layers = torch.nn.ModuleList()
        for in_size, out_size in zip(self.mlp_layer_dims[:-1], self.mlp_layer_dims[1:]):
            self.mlp_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(self.latent_dim * 2, 1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding_GMF = self.embedding_user_GMF(user_indices)
        item_embedding_GMF = self.embedding_item_GMF(item_indices)
        gmf_product = torch.mul(user_embedding_GMF, item_embedding_GMF)

        user_embedding_MLP = self.embedding_user_MLP(user_indices)
        item_embedding_MLP = self.embedding_item_MLP(item_indices)
        mlp_product = torch.cat([user_embedding_MLP, item_embedding_MLP], dim=1)
        for layer in self.mlp_layers:
            mlp_product = layer(mlp_product)
            mlp_product = torch.nn.ReLU()(mlp_product)

        ncf_product = torch.cat([gmf_product, mlp_product], dim=1)
        logits = self.affine_output(ncf_product)
        rating = self.logistic(logits)
        return rating
