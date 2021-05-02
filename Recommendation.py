import pandas as pd


def recommend(train_data):
    train_data_grouped = train_data.groupby('ProductId').agg({'UserId': 'count'}).reset_index()
    train_data_grouped.rename(columns={'UserId': 'score'}, inplace=True)
    train_data_sort = train_data_grouped.sort_values(['score', 'ProductId'], ascending=[0, 1])

    train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first')
    popularity_recommendations = train_data_sort.head(5)

    return popularity_recommendations


def recommend_items(userID, pivot_df, preds_df, num_recommendations):
    user_idx = userID - 1

    sorted_user_ratings = pivot_df.iloc[user_idx].sort_values(ascending=False)
    sorted_user_predictions = preds_df.iloc[user_idx].sort_values(ascending=False)

    temp = pd.concat([sorted_user_ratings, sorted_user_predictions], axis=1)
    temp.index.name = 'Products Recommended'
    temp.columns = ['Score', 'Prediction']

    temp = temp.loc[temp.Score == 0]
    temp = temp.sort_values('Prediction', ascending=False)
    print('\nBelow are the recommended items for user {}:'.format(userID))
    print(temp.head(num_recommendations))