from sklearn.model_selection import train_test_split

from model.itemCF import ItemCF
from model.svdMF import SvdMF
from model.userCF import UserCF
from tool.data_reader import all_ratings
from tool.log_helper import get_logger
from tool.path_helper import ROOT_DIR

ratings_train, ratings_test = train_test_split(all_ratings, random_state=42, train_size=0.9999)
for model in [
    ItemCF(knn_k=10),
    UserCF(knn_k=100),
    SvdMF(pac_p=0.9)
]:

    model_name = model.__class__.__name__
    logger = get_logger(model_name)
    logger.info("training model: " + model_name)
    model.fit(ratings_train)
    logger.info("model trained: " + model_name)

    total_error = 0
    path = ROOT_DIR.joinpath("out/" + logger.name + ".csv")
    with open(path, 'w', encoding="utf-8") as f:
        for i, r in enumerate(ratings_test):
            try:
                rating_pre = model.predict(r.user_id, r.movie_id)
                f.write("%s %s %s %s\n" % (r.user_id, r.movie_id, rating_pre, r.rating))

                # filter nan.
                if not rating_pre > 0:
                    continue

                total_error += (rating_pre - r.rating) ** 2
                mse = total_error / (i + 1)

                # log progress.
                if (i + 1) % 100 == 0:
                    f.flush()
                    logger.info("%s %s:%s %.3f " % (model_name, i + 1, len(ratings_test), mse))
            except:
                logger.exception("error for rating " + str(i))

for file_name in [
    "out/SvdMF",
    "out/ItemCF",
    "out/UserCF",
]:
    total_error = 0
    in_path = ROOT_DIR.joinpath(file_name + '.csv')
    out_path = ROOT_DIR.joinpath(file_name + ".mse")
    with open(in_path, 'r', encoding='utf-8') as f_in, \
            open(out_path, 'w', encoding='utf-8') as f_out:
        i = 0
        line = f_in.readline()
        while line:
            cols = line.strip().split(' ')
            pre = float(cols[2])
            if not pre > 0:
                continue
            i += 1
            total_error += (float(cols[3]) - pre) ** 2
            f_out.write("%s\n" % (total_error / i))
            line = f_in.readline()
