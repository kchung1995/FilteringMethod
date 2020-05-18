import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('./other_dataset/dataset2.csv')
# print(data.head(10))
# print(data.shape)

data = data.transpose()
# print(data.head(5))

data_sim = cosine_similarity(data,data)
# print(data_sim.shape)

data_sim_df = pd.DataFrame(data = data_sim, index = data.index, columns = data.index)
# print(data_sim_df.head(30))

print(data_sim_df["떡볶이"].sort_values(ascending=False)[1:10])

# https://github.com/lsjsj92/recommender_system_with_Python/blob/master/003.%20recommender%20system%20basic%20with%20Python%20-%202%20Collaborative%20Filtering.ipynb