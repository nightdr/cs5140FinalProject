import numpy as np

from data_processing import get_clean_review_df
from sklearn.metrics.pairwise import cosine_similarity


def get_most_similar_reviews(review_index, review_embeddings, n_similar_reviews):
    this_review_embedding = review_embeddings.pop(review_index)

    embedding_similarities = dict()
    for i, embedding in enumerate(review_embeddings):
        embedding_similarities[i] = cosine_similarity(np.array([this_review_embedding]), np.array([embedding]))[0][0]

    # sort by descending similarities
    embedding_similarities = dict(sorted(embedding_similarities.items(), key=lambda item: item[1], reverse=True))

    # return the indices of the top n_similar_reviews
    return list(embedding_similarities.keys())[:n_similar_reviews]


if __name__ == '__main__':
    reviews = list(np.loadtxt(r"10k_seed10_review_embeddings.csv"))
    full_df = get_clean_review_df()
    indices = full_df["Id"].to_numpy()

    most_similar_indices = get_most_similar_reviews(0, reviews, 3)
    print("Original product:")
    print(full_df.iloc[0])

    print("\nRecommended products:")
    for i in most_similar_indices:
        print(full_df.iloc[i])
        print()

