import matplotlib.pyplot as plt
from sklearn import datasets
from GMM import GMM
from scatterplot import TSNE

if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    label = iris.target
    gmm = GMM(X).gmm_result
    tsne = TSNE(X, gmm)
    layout_1 = tsne.layout()
    layout_2 = tsne.layout()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7.5))

    x_min, x_max = layout_1['x'].min() - .5, layout_1['x'].max() + .5
    y_min, y_max = layout_1['y'].min() - .5, layout_1['y'].max() + .5
    ax1.scatter(layout_1['x'], layout_1['y'], c=label, cmap=plt.cm.Set1, edgecolor='k')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)

    x_min, x_max = layout_2['x'].min() - .5, layout_2['x'].max() + .5
    y_min, y_max = layout_2['y'].min() - .5, layout_2['y'].max() + .5
    ax2.scatter(layout_2['x'], layout_2['y'], c=label, cmap=plt.cm.Set1, edgecolor='k')
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)

    fig.savefig('figure.png')