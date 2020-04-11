from dataclasses import dataclass
from typing import List

import numpy as np

from tool.log_helper import get_logger
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


def read_movies() -> List[Movie]:
    movies = []
    path = ROOT_DIR.joinpath("data/movies.dat")
    with open(path, encoding="utf-8") as f:
        for line in f.readlines():
            cols = line.strip().split("::")
            m_id = int(cols[0])
            name = cols[1][:-6].strip()
            year = int(cols[1][-5:-1])
            genres = cols[2].split("|")
            movies.append(Movie(m_id, name, year, genres))
    return movies


def read_users() -> List[User]:
    users = []
    path = ROOT_DIR.joinpath("data/users.dat")
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


def read_ratings() -> List[Rating]:
    ratings = []
    path = ROOT_DIR.joinpath("data/ratings.dat")
    with open(path, encoding="utf-8") as f:
        for line in f.readlines():
            cols = line.strip().split("::")
            user_id = int(cols[0])
            movie_id = int(cols[1])
            rating = int(cols[2])
            timestamp = int(cols[3])
            ratings.append(Rating(user_id, movie_id, rating, timestamp))
    return ratings


def generate_rating_matrix(ratings: List[Rating]) -> np.ndarray:
    """
    Generate rating matrix from ratings.
    :param ratings: a list of ratings.
    :return: the rating matrix in [user_id : movie_id], 0 for no ratings.
    """
    m = max_user_id
    n = max_movie_id
    rating_matrix = np.zeros([m, n])
    for r in ratings:
        rating_matrix[r.user_id - 1, r.movie_id - 1] = r.rating
    return rating_matrix


logger = get_logger()
logger.info("loading data from file...")
all_movies = read_movies()
all_users = read_users()
all_ratings = read_ratings()
max_movie_id = max(all_movies, key=lambda m: m.id).id
max_user_id = max(all_users, key=lambda u: u.id).id
logger.info("data loaded!")
