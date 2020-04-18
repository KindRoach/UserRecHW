from model.neuralCF import gmf, mlp
from model.neuralCF.neural_helper import load_model
from tool.data_reader import generate_implicit_ratings_with_negative, all_ratings, max_user_id, max_movie_id, \
    save_ratings, read_ratings
from tool.path_helper import ROOT_DIR

# ratings = generate_implicit_ratings_with_negative(all_ratings, negative_num=4)
# save_ratings(ratings, "data/ratings_with_negative.dat")

ratings = read_ratings("data/ratings_with_negative.dat")

config = gmf.get_default_config()
# config.use_cuda = True

model = gmf.GMF(max_user_id, max_movie_id, train_config=config)
model.fit(ratings)

# model = mlp.MLP(max_user_id, max_movie_id)
# model.fit(ratings)

# model = load_model(ROOT_DIR.joinpath("model/neuralCF/checkpoints/GMF_20200418152420_1024_0.01_0.01.pt"))
# print(model)
# for r in ratings:
#     print("%s : %s" % (r.rating, model.predict(r.user_id, r.movie_id)))
