import MisraGries
import Recommendation
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
import warnings
warnings.simplefilter('ignore')


df = pd.read_csv('Data/Food_Reviews.csv')
df = df.drop(['Id', 'ProfileName', 'Time', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Text', 'Summary',
              'ProductId', 'UserId'], axis=1)
stream = df.Score.to_list()

L, counter = MisraGries.misra_gries(stream)
print('Final Counter for Stream is :', counter)
print('Final location/label for Stream is :', L)


df = pd.read_csv('Data/Food_Reviews.csv')
df = df.drop(['Id', 'ProfileName', 'Time', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Text', 'Summary'], axis=1)
rows, columns = df.shape
print("No of rows: ", rows)
print("No of columns: ", columns)

df[['Score']].describe().transpose()
with sns.axes_style('white'):
    g = sns.factorplot("Score", data=df, aspect=2.0, kind='count')
    g.set_ylabels("Rating Count")
    g.savefig('Data/count.png')

counts = df['UserId'].value_counts()
df_final = df[df['UserId'].isin(counts[counts >= 20].index)]
print('Number of users who have rated 20 or more items =', len(df_final))
print('Number of unique USERS in final data = ', df_final['UserId'].nunique())
print('Number of unique ITEMS in final data = ', df_final['ProductId'].nunique())
stream = df_final.Score.to_list()

L, counter = MisraGries.misra_gries(stream)
print('Final Counter for Stream is :', counter)
print('Final location/label for Stream is :', L)

train_data, test_data = train_test_split(df_final, test_size = 0.3, random_state=0)
print("Test data shape: ", test_data.shape)
print("Train data shape: ", train_data.shape)


print("The top 5 popular recommendations:")
print(Recommendation.recommend(df_final))

df_CF = pd.concat([train_data, test_data]).reset_index()

p_df = pd.pivot_table(df_CF, index=['UserId'], columns='ProductId', values="Score")
p_df.fillna(0, inplace=True)
p_df['user_index'] = np.arange(0, p_df.shape[0], 1)
p_df.set_index(['user_index'], inplace=True)

# SVD
U, sigma, Vt = svds(p_df, k=20)
sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
# Predicted ratings
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = p_df.columns)

userID = 122
num_recommendations = 5
Recommendation.recommend_items(userID, p_df, preds_df, num_recommendations)
