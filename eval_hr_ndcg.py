import itertools
import math
from typing import List

from model.neuralCF.neural_helper import load_model
from tool.data_reader import max_user_id, read_ratings
from tool.log_helper import logger
from tool.path_helper import ROOT_DIR


def cal_ndcg(pred_score: List[int], k: int) -> float:
    # Local function.
    def cal_dcg(pred_score: List[int], k: int):
        dcg = 0.
        assert k <= len(pred_score)
        for i, reli in enumerate(pred_score[:k]):
            dcg += (2 ** reli - 1) / math.log(i + 2, 2)
        return dcg

    dcg = cal_dcg(pred_score, k)
    idcg = cal_dcg(sorted(pred_score, reverse=True), k)
    return dcg / idcg


def cal_hr(pred_score: List[int], k: int) -> float:
    total = sum(pred_score)
    hit = 0.
    assert k <= len(pred_score)
    for score in pred_score[:k]:
        if score == 1:
            hit += 1
    return hit / total


ratings_test = read_ratings("data/ratings_test_with_negative.dat")
gmf = load_model("model/neuralCF/checkpoints/GMF_20200419214405_1024_0.01_0.pt")
mlp = load_model("model/neuralCF/checkpoints/MLP_20200424151759_256_0.01_1e-07.pt")
nmf = load_model("model/neuralCF/checkpoints/NeuralMF_20200425012115_1024_0.01_0.pt")

for model in [gmf, mlp, nmf]:
    model_name = model.__class__.__name__

    # calculate and log hr & ndcg@10
    path = ROOT_DIR.joinpath("out/" + model_name + ".hr")
    with open(path, 'w', encoding="utf-8") as f:
        f.write("hr@10,ndcg@10\n")
        count, hr, ndcg = 0, 0., 0.
        logger.info("eval %s..." % model_name)
        for user_id, rs in itertools.groupby(ratings_test, key=lambda r: r.user_id):
            count += 1
            rs = sorted(rs, key=lambda r: model.predict(r.user_id, r.movie_id), reverse=True)
            pred = [r.rating for r in rs]
            hr += cal_hr(pred, 10)
            ndcg += cal_ndcg(pred, 10)
            f.write("%s,%s\n" % (hr / count, ndcg / count))
            if user_id % 100 == 0:
                logger.info("finish user_id %s (%.1f%%)" % (user_id, 100.0 * user_id / max_user_id))
        logger.info("eval done!")

    # log loss per epoch
    path = ROOT_DIR.joinpath("out/" + model_name + ".loss")
    with open(path, 'w', encoding="utf-8") as f:
        f.write("epoch,loss\n")
        for k, v in model.train_loss.items():
            f.write("%s,%s\n" % (k, v))
