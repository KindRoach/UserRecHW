from sklearn.model_selection import train_test_split

from model.svdMF import SvdMF
from model.userCF import UserCF
from tool.data_reader import all_ratings
from model.itemCF import ItemCF
from tool.log_helper import get_logger

ratings_train, ratings_test = train_test_split(all_ratings, random_state=42, train_size=0.8)
for model in [
    ItemCF(ratings_train, knn_k=10),
    UserCF(ratings_train, knn_k=100),
    SvdMF(ratings_train, pac_p=0.9)
]:

    model_name = model.__class__.__name__
    logger = get_logger(model_name)
    logger.info("training model: " + model_name)
    model.train()
    logger.info("model trained: " + model_name)

    total_error = 0
    with open("out/" + logger.name + ".csv", 'w', encoding="utf-8") as f:
        for i, r in enumerate(ratings_test):
            try:
                rating_pre = model.predict(r.user_id, r.movie_id)
                f.write("%s %s %s\n" % (r.user_id, r.movie_id, rating_pre))

                # filter nan.
                if not rating_pre > 0:
                    continue

                total_error += (rating_pre - r.rating) ** 2
                mse = total_error / (i + 1)

                # log progress.
                if (i + 1) % 100 == 0:
                    f.flush()
                    logger.info("%s %s:%s %.3f k=%s " % (
                        model_name,
                        i + 1, len(ratings_test),
                        mse, model.PAC_K,
                    ))
            except:
                logger.exception("error for rating " + str(i))
