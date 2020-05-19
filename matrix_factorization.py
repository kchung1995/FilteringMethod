from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('./other_dataset/dataset3.csv')

food_user_rating = data.values.T

#이하 SVD: Singular Value Decomposition, 특이값 분해 사용)
SVD = TruncatedSVD(n_components=12)
#latent 값을 10으로 둠. 이 값을 어떻게 두느냐에 따라 추천이 완전히 달라짐.
matrix = SVD.fit_transform(food_user_rating)
#print(matrix.shape)
#print(matrix[0])
#10개의 component로 차원을 축소하였음
#이를 이용하여 피어슨 상관계수 (Pearson correlation coefficient)를 구함

corr = np.corrcoef(matrix)
#print(corr.shape)
corr2 = corr[:10, :10]
#print(corr2.shape)

plt.figure(figsize = (16, 10))
sns.heatmap(corr2)
plt.show()
#첨부한 이미지대로 상관계수의 관계를 보임

#이렇게 구한 상관계수를 이용하여 특정 음식과 상관계수가 높은 음식을 뽑음

food_name = data.columns
food_name_list = list(food_name)

#곱창과 유사한 음식을 n개 출력
coffey_hands = food_name_list.index("삼계탕")
corr_coffey_hands = corr[coffey_hands]
print(list(food_name[(corr_coffey_hands >= 0.91)])[:10])
