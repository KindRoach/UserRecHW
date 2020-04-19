from model.neuralCF import gmf, mlp, nmf
from tool.data_reader import generate_implicit_ratings_with_negative, all_ratings, max_user_id, max_movie_id, \
    save_ratings, read_ratings, split_ratings_by_remain_one

ratings_train, ratings_test = split_ratings_by_remain_one(all_ratings)
ratings = generate_implicit_ratings_with_negative(ratings_train, negative_num=4)
save_ratings(ratings, "data/ratings_train_with_negative.dat")
ratings = generate_implicit_ratings_with_negative(ratings_test, negative_num=99)
save_ratings(ratings, "data/ratings_test_with_negative.dat")

ratings = read_ratings("data/ratings_train_with_negative.dat")

model = gmf.GMF(max_user_id, max_movie_id)
model.fit(ratings)

model = mlp.MLP(max_user_id, max_movie_id)
model.fit(ratings)

model = nmf.NeuralMF(max_user_id, max_movie_id)
model.fit(ratings)
