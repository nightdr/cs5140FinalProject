from sklearn.linear_model import LinearRegression
from data_processing import get_x_and_y_dfs
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    X, Y = get_x_and_y_dfs()

    linear_model = LinearRegression()
    linear_model.fit(X, Y)

    print(linear_model.coef_)
    print(linear_model.intercept_)
    print(linear_model.score(X, Y))

    x, y = np.meshgrid(np.linspace(X["HelpfulnessFraction"].min(), X["HelpfulnessFraction"].max(), 100),
                       np.linspace(X["Time"].min(), X["Time"].max(), 100))
    graph_X = np.array([x.flatten(), y.flatten()]).T
    graph_z = linear_model.predict(graph_X)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(X["HelpfulnessFraction"], X["Time"], Y, alpha=0.5)
    ax.plot_trisurf(x.flatten(), y.flatten(), graph_z, color="r")

    ax.set_title("Customer Food Review Linear Regression")
    ax.set_xlabel("Helpfulness Percentage (0% to 100%)")
    ax.set_ylabel("Time (epoch ms)")
    ax.set_zlabel("Product Rating (1-5)")
    plt.show()
    fig.savefig("linearRegression.png", bbox_inches="tight")
