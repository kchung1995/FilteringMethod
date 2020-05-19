#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from surprise import Dataset
from surprise import SVD
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate


ratings = pd.read_csv('./movie_dataset/ratings.csv', usecols = ['userId', 'movieId', 'rating'])
reader = Reader(rating_scale=(1, 5))

ratings = Dataset.load_from_df(ratings, reader)

raw_ratings = ratings.raw_ratings

random.Random(10).shuffle(raw_ratings)

threshold = int(.9 * len(raw_ratings))
A_raw_ratings = raw_ratings[:threshold]
B_raw_ratings = raw_ratings[threshold:]

ratings.raw_ratings = A_raw_ratings

train_set = ratings.build_full_trainset()
test_set = ratings.construct_testset(B_raw_ratings)

algo = SVD()
cross_val = cross_validate(algo, ratings, measures=['RMSE', 'MAE'], cv=5, verbose=True)

algo = SVD()
algo.fit(train_set)

train_svd = algo.test(train_set.build_testset())
rmse = accuracy.rmse(train_svd)
mae = accuracy.mae(train_svd)

test_svd = algo.test(test_set)
rmse = accuracy.rmse(test_svd)
mae = accuracy.mae(test_svd)