import itertools
import random
from dataclasses import dataclass
from typing import List

import numpy as np

from tool.log_helper import logger
from tool.path_helper import ROOT_DIR


@dataclass
class Movie(object):
    id: int
    name: str
    year: int
    genres: List[str]


@dataclass
class User(object):
    id: int
    gender: str
    age: int
    occupation: int
    zip_code: str


@dataclass
class Rating(object):
    user_id: int
    movie_id: int
    rating: int
    timestamp: int


def read_movies(path="data/movies.dat") -> List[Movie]:
    movies = []
    path = ROOT_DIR.joinpath(path)
    with open(path, encoding="utf-8") as f:
        for line in f.readlines():
            cols = line.strip().split("::")
            m_id = int(cols[0])
            name = cols[1][:-6].strip()
            year = int(cols[1][-5:-1])
            genres = cols[2].split("|")
            movies.append(Movie(m_id, name, year, genres))
    return movies


def read_users(path="data/users.dat") -> List[User]:
    users = []
    path = ROOT_DIR.joinpath(path)
    with open(path, encoding="utf-8") as f:
        for line in f.readlines():
            cols = line.strip().split("::")
            user_id = int(cols[0])
            gender = cols[1]
            age = int(cols[2])
            occupation = int(cols[3])
            zip_code = cols[4]
            users.append(User(user_id, gender, age, occupation, zip_code))
    return users


def read_ratings(path="data/ratings.dat") -> List[Rating]:
    ratings = []
    path = ROOT_DIR.joinpath(path)
    with open(path, encoding="utf-8") as f:
        for line in f.readlines():
            cols = line.strip().split("::")
            user_id = int(cols[0])
            movie_id = int(cols[1])
            rating = int(cols[2])
            timestamp = int(cols[3])
            ratings.append(Rating(user_id, movie_id, rating, timestamp))
    return ratings


def save_ratings(ratings: List[Rating], path: str):
    path = ROOT_DIR.joinpath(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in ratings:
            f.write("%d::%d::%d::%d\n" % (r.user_id, r.movie_id, r.rating, r.timestamp))


def generate_rating_matrix(ratings: List[Rating]) -> np.ndarray:
    """
    Generate rating matrix from ratings.
    :param ratings: a list of ratings.
    :return: the rating matrix in [user_id : movie_id], 0 for no ratings.
    """
    # id starts from 1.
    rating_matrix = np.zeros([max_user_id + 1, max_movie_id + 1])
    for r in ratings:
        rating_matrix[r.user_id, r.movie_id] = r.rating
    return rating_matrix


def generate_implicit_ratings_with_negative(ratings: List[Rating], negative_num: int = 99) -> List[Rating]:
    logger.info("generating ratings with negative samples...")
    # id starts from 1
    all_movies_id = set(range(1, max_movie_id + 1))
    ratings_with_negative = []
    random.seed(42)
    for user_id, rs in itertools.groupby(ratings, key=lambda r: r.user_id):
        # covert rs to list for using twice.
        rs = list(rs)
        interacted_movies = set(r.movie_id for r in rs)
        negative_movies = all_movies_id - interacted_movies
        for r in rs:
            negative_samples = random.sample(negative_movies, negative_num)
            negative_ratings = [Rating(user_id, movie_id, 0, r.timestamp) for movie_id in negative_samples]
            # convert rating score to binary value, 1 means interacted.
            ratings_with_negative.append(Rating(r.user_id, r.movie_id, 1, r.timestamp))
            # every positive ratings follow by {negative_num} negative ratings.
            ratings_with_negative.extend(negative_ratings)

        if user_id % 100 == 0:
            logger.info("finish user_id %s (%.1f%%)" % (user_id, 100.0 * user_id / max_user_id))

    logger.info("ratings generated!")
    return ratings_with_negative


logger.info("loading origin data from file...")
all_movies = read_movies()
all_users = read_users()
all_ratings = read_ratings()
max_movie_id = max(all_movies, key=lambda m: m.id).id
max_user_id = max(all_users, key=lambda u: u.id).id
logger.info("data loaded!")
