from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from data_processing import get_clean_review_df

if __name__ == '__main__':
    reviews = np.loadtxt(r"10k_seed10_review_embeddings.csv")
    full_df = get_clean_review_df()
    ratings = full_df["Score"].to_numpy()

    PCA_model = PCA(n_components=3)
    PCA_reviews = PCA_model.fit_transform(reviews)

    # get clusters
    clusters = defaultdict(list)
    for index in range(len(reviews)):
        clusters[ratings[index]].append(PCA_reviews[index])
    ratings = clusters.keys()
    clusters = clusters.values()

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    for cluster in clusters:
        ax.scatter(*np.array(cluster).T)
    ax.set_title("Customer Food Review Rating Clusters")
    ax.set_xlabel("First Principle Component")
    ax.set_ylabel("Second Principle Component")
    ax.set_zlabel("Third Principle Component")
    if len(clusters) > 1:
        legend_labels = []
        for i in range(len(clusters)):
            legend_labels.append("Rating = " + str(i))
        ax.legend(legend_labels, loc="upper left")
    plt.show()
    fig.savefig("clustering.png", bbox_inches="tight")