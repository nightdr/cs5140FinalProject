import numpy as np
from sentence_transformers import SentenceTransformer
from data_processing import get_clean_review_df

if __name__ == '__main__':
    review_df = get_clean_review_df()

    bert_model = SentenceTransformer("bert-base-nli-mean-tokens")

    reviews = review_df["Text"].to_numpy()

    print("Starting encoding reviews")
    encoded_reviews = np.array(bert_model.encode(reviews, show_progress_bar=True))
    print("Finished encoding reviews")

    print("\nStarted saving reviews")
    np.savetxt(r"C:\Users\madma\Desktop\review_embeddings.csv", encoded_reviews)
    print("Finished saving reviews")
