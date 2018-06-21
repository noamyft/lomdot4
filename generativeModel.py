from sklearn.naive_bayes import GaussianNB

import stats
import pandas as pd
import numpy as np

def getMeanStdWithBayes(x, y):
    clf = GaussianNB()
    clf.fit(x, y)
    print(clf.score(x,y))

    # print(clf.classes_.shape)
    # print(clf.classes_)
    # print(x.columns)
    # print(clf.theta_.shape)
    # print(clf.theta_)
    # print(clf.sigma_.shape)
    # print(clf.sigma_)

    res = {}
    for i,cls in enumerate(clf.classes_):
        res[cls] = list(zip(x.columns, clf.theta_[i], clf.sigma_[i]))


    # print mead,std for each class
    for clas, meanstd in res.items():
        print("stats for: " + clas)
        for r in meanstd:
            if r[2] <= 0.1:
                print(r[0:3])

    meanDifferences = calcDistances(clf.classes_.tolist(), clf.theta_)

    print(meanDifferences)
    stats.plotPCA(X=pd.DataFrame(clf.theta_), Y=pd.Series(clf.classes_), title="PCA - RIGHT FEATURES",
                  normalize="none", n=2)
    stats.plotPCA(X=pd.DataFrame(clf.theta_), Y=pd.Series(clf.classes_), title="PCA - RIGHT FEATURES",
                  normalize="min-max", n=2)


def calcDistances(classesNames, classesVectors) -> pd.DataFrame:
    """
    compute the distance between vectors

    :param classesNames: list with size of [n] with the name of each vector
    :param classesVectors: [n,m] array with all the vectors to calculate differences
    (n-num of vectors to compare, m - size of vector)
    :return dataFrame with distances
    """
    # print mean diffrences
    meanDifferences = pd.DataFrame(columns=["class"] + classesNames)
    meanDifferences["class"] = classesNames
    for i, cls in enumerate(classesNames):
        #     calculate l2 norm for cls
        d = np.sum((classesVectors - classesVectors[i]) ** 2, axis=1).reshape([1, -1])
        meanDifferences.iloc[i, 1:] = d
    return meanDifferences