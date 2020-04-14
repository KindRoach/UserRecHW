from model.neuralCF import gmf
from tool.data_reader import generate_implicit_ratings_with_negative, all_ratings, max_user_id, max_movie_id, \
    save_ratings, read_ratings

# ratings = generate_implicit_ratings_with_negative(all_ratings, negative_num=4)
# save_ratings(ratings, "data/ratings_with_negative.dat")
ratings = read_ratings("data/ratings_with_negative.dat")
model = gmf.GMF(max_user_id, max_movie_id)
config = gmf.default_config
config.batch_size = 16 * 1024
model.fit(ratings)
