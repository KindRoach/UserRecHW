from model.neuralCF import gmf, mlp
from tool.data_reader import generate_implicit_ratings_with_negative, all_ratings, max_user_id, max_movie_id, \
    save_ratings, read_ratings

ratings = generate_implicit_ratings_with_negative(all_ratings, negative_num=4)
save_ratings(ratings, "data/ratings_with_negative.dat")
# ratings = read_ratings("data/ratings_with_negative.dat")

model = gmf.GMF(max_user_id, max_movie_id)
model.fit(ratings)

model = mlp.MLP(max_user_id, max_movie_id)
model.fit(ratings)
