import torch

from model.neuralCF.neural_helper import TrainConfig, BaseModel


def get_default_config() -> TrainConfig:
    return TrainConfig(num_epochs=100,
                       batch_size=1024,
                       learning_rate=0.01,
                       l2_regularization=0.0,
                       use_cuda=True)


class GMF(BaseModel):
    def __init__(self, num_users: int, num_items: int, latent_dim: int = 8, train_config: TrainConfig = None):
        if train_config is None:
            train_config = get_default_config()
        super(GMF, self).__init__(train_config)

        self.num_users = num_users + 1
        self.num_items = num_items + 1
        self.latent_dim = latent_dim
        self.embedding_user = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_item = torch.nn.Embedding(self.num_items, self.latent_dim)

        self.affine_output = torch.nn.Linear(self.latent_dim, 1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        rating = self.logistic(logits)
        return rating
