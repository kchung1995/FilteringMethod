from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

ratings = pd.read_csv('./movie_dataset/ratings.csv')
foods = pd.read_csv('./movie_dataset/movies.csv')

user_food_ratings = ratings.pivot(
    index = 'userId',
    columns = 'movieId',
    values = 'rating'
).fillna(0)

#print(user_food_ratings.head())

matrix = user_food_ratings.values
user_ratings_mean = np.mean(matrix,axis=1)
matrix_user_mean = matrix - user_ratings_mean.reshape(-1,1)

#print(matrix)

#print(pd.DataFrame(matrix_user_mean, columns = user_food_ratings.columns).head())

U, sigma, Vt = svds(matrix_user_mean, k = 12)
#print(U.shape)
#print(sigma.shape)
#print(Vt.shape)

sigma = np.diag(sigma)
#대칭 행렬로 변환

svd_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1,1)

svd_preds = pd.DataFrame(matrix_user_mean, columns = user_food_ratings.columns)


#함수 제작
def recommend_foods(svd_preds, user_id, ori_foods, ori_ratings, num_recommendations = 5):
    user_row_number = user_id-1
    sorted_user_predictions = svd_preds.iloc[user_row_number].sort_values(ascending=False)
    user_data = ori_ratings[ori_ratings.userId == user_id]
    user_history = user_data.merge(ori_foods, on = 'movieId').sort_values(['rating'], ascending = False)
    recommendations = ori_foods[~ori_foods['movieId'].isin(user_history['movieId'])]
    recommendations = recommendations.merge(pd.DataFrame(sorted_user_predictions).reset_index(), on = 'movieId')
    recommendations = recommendations.rename(columns = {user_row_number: 'Predictions'}).sort_values('Predictions', ascending = False).iloc[:num_recommendations, :]

    return user_history, recommendations

already_rated, predictions = recommend_foods(svd_preds, 178, foods, ratings, 10)

pd.set_option('display.max_columns', None)
print(already_rated.head(10))
print(predictions)