import pandas as pd

from clustering import clusterDistribution


def changeAndItsPrice(x_train,y_train):

    # changing factors
    x_train.loc[(x_train['Number_of_valued_Kneset_membersFillByMedian'] > 0.5) &
                (x_train['Will_vote_only_large_partyFillByMode_No'] > 0.5) &
                (x_train['Overall_happiness_scoreFillByMean'] < 1.083),'Number_of_valued_Kneset_membersFillByMedian'] *= 1000
    # x_train.loc[:,'Garden_sqr_meter_per_person_in_residancy_areaFillByMean'] = 1
    # x_train.loc[:,'Will_vote_only_large_partyFillByMode_No'] = 0
    # x_train.loc[:,'Will_vote_only_large_partyFillByMode_Yes'] = 0
    # x_train.loc[:,'Weighted_education_rankFillByMean'] *= 10
    # x_train.loc[:, 'Overall_happiness_scoreFillByMean'] *= 10
    # x_train.loc[:, 'Yearly_IncomeKFillByMean'] = 2
    # x_train.loc[:, 'Avg_Satisfaction_with_previous_voteFillByMean']


    for i in range(2,8):
        clusterDistribution(x_train, Y=y_train, k=i,clusModel='kmeans')
