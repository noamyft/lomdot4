import numpy as np
import pandas as pd
from sklearn.cluster import KMeans,SpectralClustering,AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

import stats
from generativeModel import calcDistances



colPool = [ '#bd2309', '#bbb12d', '#1480fa', '#14fa2f', '#000000',\
                     '#faf214', '#2edfea', '#ea2ec4', '#ea2e40', '#cdcdcd',\
                    '#577a4d', '#2e46c0', '#f59422', '#219774', '#8086d9' ]

def plotKmeans(X, Y, k=3):

    kmeans = KMeans(n_clusters=k)
    y_kmeans = kmeans.fit_predict(X)
    Y['kmeans'] = y_kmeans
    print(Y)



    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    # plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

    Y = Y.values
    labels_with_colors = {"Blues" : "b",
                          "Browns" : "brown",
                          "Purples": "purple",
                          "Whites" : "k",
                          "Pinks": "pink",
                          "Turquoises": "c",
                          "Oranges": "orange",
                          "Yellows": "yellow",
                          "Greens": "g",
                          "Greys": "grey",
                          "Reds": "r"}

    for label, color in labels_with_colors.items():
        labelIndexes = np.where(Y == label)
        # plt.subplot(3,4,i)
        plt.scatter(X[labelIndexes, 0], X[labelIndexes, 1], c=color,
                    label=label)

    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()



def clusterDistribution(X, Y, k=3, clusModel ='kmeans'):
    """

    :param X:
    :param Y:
    :param k:
    :param clusModel:
    :return:
    """
    if clusModel == 'kmeans':
        clf = kmeans = KMeans(n_clusters=k)
        y_pred = kmeans.fit_predict(X)
    elif clusModel == 'GMM':
        clf = gmm = GaussianMixture(n_components=k)
        gmm.fit(X)
        y_pred = gmm.predict(X)
    elif clusModel == 'Spec':
        clf = spec = SpectralClustering(k)
        y_pred = spec.fit_predict(X)
    elif clusModel == 'Agg':
        clf = agg = AgglomerativeClustering(k)
        y_pred = agg.fit_predict(X)

    Y[clusModel] = y_pred

    clusterSize = { str(cluster):round(Y.loc[Y[clusModel]==cluster].shape[0]/Y.shape[0],2)
                    for cluster in range(k)}

    new_plot = pd.crosstab([Y.Vote], Y[clusModel])
    new_plot.plot(kind='bar', stacked=True, \
                  color=colPool, grid=False)
    title = "Distribution of {} in different parties"
    plt.title(title.format(clusModel+str(k)))
    plt.xlabel('Name of Party')
    plt.ylabel('Number of Voters')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    title += '.png'
    legend1 = plt.text(x=0,y=1750,s=clusterSize)

    plotName = title.format(clusModel+str(k))
    plt.savefig("plot/" + plotName, bbox_inches="tight")
    plt.clf()

    # print distances between clusters
    print("distances between clusters for ", clusModel + str(k))
    print(calcDistances(["cluster " + str(i) for i in range (0,k)], clf.cluster_centers_).to_string())

    # plot clusters
    labels_with_colors = np.array(["Blues", "Browns", "Purples", "Whites", "Pinks","Turquoises","Oranges","Yellows",
                          "Greens", "Greys", "Reds"]*3)
    stats.plotReductionDims(X=pd.DataFrame(X), Y=pd.Series(labels_with_colors[y_pred]),
                  title="scatter cluster of "+clusModel + str(k), normalize="none",method="tsne",
                            n=2, toShow=False, toSave= True)
