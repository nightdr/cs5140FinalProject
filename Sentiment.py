import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def predict(model, vec, statement):
    print('Checking if the following statement review is positive or negative:')
    print(statement)
    result = model.predict(vec.transform(statement))
    if result == 1:
        print('Prediction: Positive Review')
    else:
        print('Prediction: Negative Review')


df = pd.read_csv('Data/Food_Reviews.csv')
df = df.sample(100000, random_state=10)
df.dropna(inplace=True)
df['Positivity'] = np.where(df['Score'] > 3, 1, 0)
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Positivity'], random_state=10)

vec = CountVectorizer(min_df=5, ngram_range=(1, 2)).fit(X_train)
X_train_vectorized = vec.transform(X_train)
len(vec.get_feature_names())
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
predictions = model.predict(vec.transform(X_test))
print('Area Under Curve score on testing set predictions: ', roc_auc_score(y_test, predictions))
feature_names = np.array(vec.get_feature_names())
sorted_coef_index = model.coef_[0].argsort()
print('Most negative coefficient words: \n{}\n'.format(feature_names[sorted_coef_index][:10]))
print('Most positive coefficient words: \n{}\n'.format(feature_names[sorted_coef_index][:-11:-1]))
predict(model, vec, ['The product is not good, I would never buy them again'])
predict(model, vec, ['The product is not bad, I will buy them again'])
