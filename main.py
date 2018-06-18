import numpy as np
import pandas as pd
from PyQt5.QtCore import center
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import tree
from sklearn.cluster import KMeans



import stats
from TreePlot import plotTree

import mainPreprocessing
from clustering import plotKmeans, clusterDistribution


def chooseColumns(currCols,oldCols):
    modeSuffix = 'FillByModeInt'
    onehotSuffix = 'FillByMode_'
    meanSuffix = 'FillByMean'
    meadianSuffix = 'FillByMedian'
    nanSuffix = '_nan'
    # currList= list of original cols name, res= set of cols to continue with
    res = [colName for colName in currCols if onehotSuffix in colName]
    res += [colName for colName in currCols if (meadianSuffix in colName) & (meanSuffix not in colName)]
    currListMode = [colName for colName in oldCols if (colName + modeSuffix) in currCols]
    res += [(colName + modeSuffix) for colName in currListMode]
    currListMean = [colName for colName in oldCols if ((colName + meanSuffix) in currCols) & (colName not in currListMode)]
    res += [(colName + meanSuffix) for colName in currListMean if (colName + meadianSuffix) not in res]
    res = [colName for colName in res if nanSuffix not in colName]
    return res


def chooseRightColumns(colsAfterWork):
    """

    :param colsAfterWork: the names of columns without nan
    :return: list with the relevant features
    """
    rightList = ["Yearly_IncomeKFillByMean" ,"Number_of_valued_Kneset_membersFillByMedian",
                 "Overall_happiness_scoreFillByMean" ,"Garden_sqr_meter_per_person_in_residancy_areaFillByMean",
                 "Most_Important_IssueFillByMode_" ,"Weighted_education_rank",
                 "Will_vote_only_large_partyFillByMode_" ,"Avg_Satisfaction_with_previous_vote"]
    res = []
    for feature in rightList:
        for oldFeat in colsAfterWork:
            if feature in oldFeat:
                res.append(oldFeat)
    return res



def loadEnvirment(onlyRightCols = True):
    """

    :return: 6 dataFrames of train, validarion and test x/y
    """

    df = pd.read_csv("./input/ElectionsData.csv")
    oldCols = df.columns

    ## load tables from prev hw
    x_train = pd.read_csv("./input/x_train.csv" ,index_col=0)
    x_val = pd.read_csv("./input/x_val.csv", index_col=0)
    x_test = pd.read_csv("./input/x_test.csv", index_col=0)
    y_train = pd.read_csv("./input/y_train.csv" ,index_col=0)
    y_val = pd.read_csv("./input/y_val.csv", index_col=0)
    y_test = pd.read_csv("./input/y_test.csv", index_col=0)


    # choose the correct set of features
    colsAfterWork = chooseColumns(x_train.columns, oldCols)
    x_train = x_train[colsAfterWork]
    x_val = x_val[colsAfterWork]
    x_test = x_test[colsAfterWork]

    if onlyRightCols:
        rightFeatures = chooseRightColumns(colsAfterWork)
        x_train = x_train[rightFeatures]
        x_val = x_val[rightFeatures]
        x_test = x_test[rightFeatures]

    return x_train, x_val, x_test, y_train, y_val, y_test




def saveEnvirment(x_train, x_val, x_test, y_train, y_val, y_test):
    # save after all changes
    x_train_final = x_train.copy()
    x_train_final['Vote'] = y_train.values
    x_val_final = x_val.copy()
    x_val_final['Vote'] = y_val.values
    x_test_final = x_test.copy()
    x_test_final['Vote'] = y_test.values
    # Save labels
    x_train_final.to_csv("./x_train_final.csv")
    x_val_final.to_csv("./x_val_final.csv")
    x_test_final.to_csv("./x_test_final.csv")
    y_train.to_csv("./y_train.csv")
    y_val.to_csv("./y_val.csv")
    y_test.to_csv("./y_test.csv")

def main():

    # # With preprocessing
    # mainPreprocessing.main()

    np.random.seed(0)

    (x_train, x_val, x_test, y_train, y_val, y_test) = loadEnvirment()
    # print("\nOnly right columns:\n")
    # x_train.info()
    (x_train_all, x_val_all, x_test_all, y_train_all, y_val_all, y_test_all) = \
        loadEnvirment(onlyRightCols=False)
    # print("\nAll columns:\n")
    # x_train_all.info(verbose=True)
    # saveEnvirment(x_train, x_val, x_test, y_train, y_val, y_test)




    # # For drawing the decision tree
    # labelsForTree = ['Blues', 'Browns', 'Greens', 'Greys', 'Oranges', 'Pinks',
    #                  'Purples', 'Reds', 'Turquoises', 'Whites', 'Yellows']
    # plotTree(estimator, x_train.columns, labelsForTree)


    # reduced_data  = stats.plotPCA(X=x_train,Y=y_train,title="PCA - RIGHT FEATURES",normalize=None)
    # reduced_data = stats.plotPCA(X=x_train, Y=y_train, title="PCA - RIGHT FEATURES", normalize="min-max")
    # stats.plotPCA(X=x_train, Y=y_train, title="PCA - RIGHT FEATURES", normalize="normal")
    #
    # stats.plotPCA(X=x_train_all, Y=y_train_all, title="PCA - ALL FEATURES",normalize=None)
    # stats.plotPCA(X=x_train_all, Y=y_train_all, title="PCA - ALL FEATURES", normalize="min-max")
    # stats.plotPCA(X=x_train_all, Y=y_train_all, title="PCA - ALL FEATURES", normalize="normal")

    # KMEANS trail
    # plotKmeans(x_train,Y=y_train)

    # plt.scatter(x_train.iloc[:,0],x_train.iloc[:,1],c=y_kmeans,s=50,cmap='viridis') # for 2-d
    # centers = k_means.cluster_centers_
    # plt.scatter(centers[:,0], centers[:,1],marker='X')
    # plt.show()
    for i in range(2,15):
        # reduced_data = stats.plotPCA(X=x_train, Y=y_train, title="PCA - RIGHT FEATURES",
        #                              n=i, normalize="min-max")
        clusterDistribution(x_train, Y=y_train, k=i,clusModel='Agg')
        # clusterDistribution(x_train, Y=y_train, k=i, clusModel='Spec')







if __name__ == '__main__':
    main()

