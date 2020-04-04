from sklearn.model_selection import train_test_split

from tool.data_reader import all_ratings
from model.itemCF import ItemCF
from tool.log_helper import get_logger

logger = get_logger()
ratings_train, ratings_test = train_test_split(all_ratings, random_state=42, train_size=0.8)
for model in [ItemCF(ratings_train, knn_n=100)]:
    logger.info("training model: " + model.__class__.__name__)
    model.train()
    logger.info("model trained: " + model.__class__.__name__)

    total_error = 0
    for i, r in enumerate(ratings_test):
        rating_pre = model.predict(r.user_id, r.movie_id)
        total_error += (rating_pre - r.rating) ** 2
        mse = total_error / (i + 1)
        if (i + 1) % 100 == 0:
            logger.info("%s %s:%s  %.3f" % (model.__class__.__name__, i + 1, len(ratings_test), mse))
