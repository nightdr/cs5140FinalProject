import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from data_processing import get_clustering_df

if __name__ == "__main__":
    # review_df = get_clustering_df()
    reviews = np.loadtxt(r"10k_seed10_review_embeddings.csv")
    k = 5
    clustering_model = KMeans(n_clusters=k)
    clustering_model.fit(reviews)

    PCA_model = PCA(n_components=3)
    PCA_reviews = PCA_model.fit_transform(reviews)

    # TODO agglomerative clustering too?

    # get clusters
    labels = clustering_model.labels_
    clusters = defaultdict(list)
    for index in range(len(labels)):
        clusters[labels[index]].append(PCA_reviews[index])
    clusters = clusters.values()

    print("done building clusters")

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    for cluster in clusters:
        ax.scatter(*np.array(cluster).T)
    ax.set_title("Customer Food Review k-Means Clusters with k={}".format(k))
    ax.set_xlabel("First Principle Component")
    ax.set_ylabel("Second Principle Component")
    ax.set_zlabel("Third Principle Component")
    # if len(clusters) > 1:
    #     legend_labels = []
    #     for i in range(len(clusters)):
    #         legend_labels.append("Cluster " + str(i))
    #     ax.legend(legend_labels, loc="upper left")
    plt.show()
    fig.savefig("clustering.png", bbox_inches="tight")
