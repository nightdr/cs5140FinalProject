import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords
import re


def get_clean_review_df():
    review_df = pd.read_csv(r"../Data/foodReviews.csv")
    review_df = review_df.sample(10000, random_state=10)
    print(review_df["Text"].to_numpy()[0])

    # add HelpfulnessFraction from HelpfulnessNumerator and HelpfulnessDenominator
    review_df["HelpfulnessFraction"] = round(review_df["HelpfulnessNumerator"] / review_df["HelpfulnessDenominator"], 4)
    review_df["HelpfulnessFraction"] = review_df["HelpfulnessFraction"].replace(np.nan, 0)
    # replace any fractions > 1 with 1
    review_df["HelpfulnessFraction"] = review_df["HelpfulnessFraction"].apply(lambda x: 1 if x > 1 else x)

    # from https://towardsdatascience.com/calculating-document-similarities-using-bert-and-other-models-b2c1a29c9630
    # clean review text by converting words to lowercase and removing special and stop characters
    stop_words = stopwords.words("english")
    review_df["Text"] = review_df["Text"].apply(lambda x: " ".join(
        re.sub(r"[^a-zA-Z]", "", w).lower() for w in x.split() if
        re.sub(r"[^a-zA-Z]", "", w).lower() not in stop_words))

    print(review_df["Text"].to_numpy()[0])

    return review_df


def get_x_and_y_dfs():
    review_df = get_clean_review_df()
    return review_df[["HelpfulnessFraction", "Time"]], review_df["Score"]


def get_clustering_df():
    review_df = get_clean_review_df()

    # scale HelpfulnessFraction, Time, and Score to (0, 1)
    min_max_scaler = MinMaxScaler()
    attributes = ["HelpfulnessFraction", "Time", "Score"]
    for attribute in attributes:
        review_df[attribute] = min_max_scaler.fit_transform(review_df[[attribute]])

    return review_df[attributes]


if __name__ == "__main__":
    print(get_clean_review_df()["Text"])
