import itertools
import math
from typing import List

from model.neuralCF.neural_helper import load_model
from tool.data_reader import split_ratings_by_remain_one, all_ratings, generate_implicit_ratings_with_negative
from tool.log_helper import logger


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


ratings_train, ratings_test = split_ratings_by_remain_one(all_ratings)
ratings = generate_implicit_ratings_with_negative(ratings_test, negative_num=99)

model = load_model("model/neuralCF/checkpoints/NeuralMF_20200418195954_1024_0.01_0.pt")

count = 0
hr = 0.
ndcg = 0.
for user_id, rs in itertools.groupby(ratings, key=lambda r: r.user_id):
    count += 1
    rs = sorted(rs, key=lambda r: model.predict(r.user_id, r.movie_id), reverse=True)
    pred = [r.rating for r in rs]
    hr += cal_hr(pred, 10)
    ndcg += cal_ndcg(pred, 10)
    logger.info("Hr@10=%.3f Ndcg@10=%.3f" % (hr / count, ndcg / count))
