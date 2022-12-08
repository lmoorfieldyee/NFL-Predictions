import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from numpy import *
import os
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

week = int(input("Enter current week's matchup: "))
week_folder = r"C:\Users\lmoor\Desktop\Data Science Projects\NFL Predictions\week " + str(week)
if len(os.listdir(week_folder)) == 0:
    get_clean_data(2022, week)
    get_clean_data(2021, week)
    get_clean_data(2020, week)
    get_clean_data(2019, week)
    get_clean_data(2018, week)

df_2022 = pd.read_csv(r"C:\Users\lmoor\Desktop\Data Science Projects\NFL Predictions\week " + str(week) + "\df_2022.csv")
df_2021 = pd.read_csv(r"C:\Users\lmoor\Desktop\Data Science Projects\NFL Predictions\week " + str(week) + "\df_2021.csv")
df_2020 = pd.read_csv(r"C:\Users\lmoor\Desktop\Data Science Projects\NFL Predictions\week " + str(week) + "\df_2020.csv")
df_2019 = pd.read_csv(r"C:\Users\lmoor\Desktop\Data Science Projects\NFL Predictions\week " + str(week) + "\df_2019.csv")
df_2018 = pd.read_csv(r"C:\Users\lmoor\Desktop\Data Science Projects\NFL Predictions\week " + str(week) + "\df_2018.csv")

all_df = pd.concat([df_2022, df_2021, df_2020, df_2019, df_2018])

train_df = all_df[(all_df['season']!=2022) & (all_df['week']==week)]
predict_df = all_df[(all_df['season']==2022) & (all_df['week']==week)]

'''
testing out which features to include in the model. many of the variables are co-related and therefore we should not 
include all of them. I.e. 'total yards scored' is the sum of 'total pass yards' and 'total rush yards' so we should 
use one or the other. For this model feature set 6 produces the best results.

feature sets roc_auc_scores are below:
features0 = 0.6677205300632911
features1 = 0.6748164556962026
features2 = 0.6838587816455696
features3 = 0.6865810917721518 (alpha=1)
features4 = 0.6883500791139241 (alpha=1)
features5 = 0.6863334651898734 (alpha=1)
features6 = 0.6884422468354431 (alpha=1)
features7 = 0.6841538765822784 (alpha=.1)
features8 = 0.6883995253164557 (alpha =.1)
features9 = 0.6856863132911393 (alpha = 1)
features10 = 0.6852167721518987 (alpha = 1)
features_all = 0.6863334651898734
'''

'''
X_features0 = ['win ratio', 'opp win ratio']

X_features1 = ['win ratio', 'opp win ratio', 'home or away']

X_features2 = ['win ratio', 'opp win ratio', 'home or away', 'total points scored szn avg', 'opp total points scored szn avg']

X_features3 = ['win ratio', 'opp win ratio', 'home or away', 'total points scored szn avg', 'opp total points scored szn avg',
              'total points allowed szn avg', 'opp total points allowed szn avg']

X_features4 = ['win ratio', 'opp win ratio', 'home or away', 'total points scored szn avg', 'opp total points scored szn avg',
              'total points allowed szn avg', 'opp total points allowed szn avg', 'total yards against szn avg', 
              'opp total yards against szn avg']

X_features5 = ['win ratio', 'opp win ratio', 'home or away', 'total points scored szn avg', 'opp total points scored szn avg',
              'total points allowed szn avg', 'opp total points allowed szn avg', 'total yards against szn avg', 
              'opp total yards against szn avg', 'total yards gained szn avg', 'opp total yards gained szn avg']



X_features6 = ['win ratio', 'opp win ratio', 'home or away', 'total points scored szn avg', 'opp total points scored szn avg',
               'total yards against szn avg', 'opp total yards against szn avg']

X_features7 = ['win ratio', 'opp win ratio', 'home or away', 'total points scored szn avg', 'opp total points scored szn avg',
               'total yards against szn avg', 'opp total yards against szn avg', 'total yards gained szn avg', 
              'opp total yards gained szn avg']

X_features8 = ['win ratio', 'opp win ratio', 'home or away', 'total points scored szn avg', 'opp total points scored szn avg',
               'total yards against szn avg', 'opp total yards against szn avg', 'first downs gained szn avg',
              'opp first downs gained szn avg']

X_features9 = ['win ratio', 'opp win ratio', 'home or away', 'total points scored szn avg', 'opp total points scored szn avg',
               'total yards against szn avg', 'opp total yards against szn avg', 'turnovers gained by defense szn avg',
               'opp turnovers gained by defense szn avg']

X_features10 = ['win ratio', 'opp win ratio', 'home or away', 'total points scored szn avg', 'opp total points scored szn avg',
               'total yards against szn avg', 'opp total yards against szn avg', 'turnovers lost szn avg',
               'opp turnovers lost szn avg']

X_features_all = ['home or away', 'win ratio',
                  'total points scored szn avg', 'total points allowed szn avg',
                  'first downs gained szn avg',
                  'total yards gained szn avg', 'pass yards gained szn avg',
                  'rush yards gained szn avg', 'turnovers lost szn avg',
                  'first downs allowed szn avg', 'total yards against szn avg',
                  'pass yards against szn avg', 'rush yards against szn avg',
                  'turnovers gained by defense szn avg',
                  'opp total points scored szn avg', 'opp total points allowed szn avg',
                  'opp first downs gained szn avg', 'opp total yards gained szn avg',
                  'opp pass yards gained szn avg', 'opp rush yards gained szn avg',
                  'opp turnovers lost szn avg', 'opp first downs allowed szn avg',
                  'opp total yards against szn avg', 'opp pass yards against szn avg',
                  'opp rush yards against szn avg',
                  'opp turnovers gained by defense szn avg', 'opp win ratio']
'''


X_features6 = ['win ratio', 'opp win ratio', 'home or away', 'total points scored szn avg', 'opp total points scored szn avg',
               'total yards against szn avg', 'opp total yards against szn avg']


X = train_df[X_features6]
y = train_df['win or loss']
X_predict = predict_df[X_features6]

#normalizing values
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X)
X_predict_scaled = scaler.transform(X_predict)

alphas = [.001, .01, .1, 1, 10, 100]

for alpha in alphas:
    print("current alpha {}".format(alpha))
    clf = LogisticRegression(max_iter=10000, penalty='l2', C=alpha)
    # clf.fit(X_train_scaled, y_train)
    results = cross_validate(estimator=clf,
                             X=X_train_scaled,
                             y=y,
                             cv=5,
                             scoring=['accuracy', 'roc_auc'],
                             return_train_score=True)

    print('training accuracy of: {}'.format(results['train_accuracy'].mean()))
    print('test accuracy of: {}'.format(results['test_accuracy'].mean()))
    print('training roc: {}'.format(results['train_roc_auc'].mean()))
    print('test roc: {}'.format(results['test_roc_auc'].mean()))
    print('\n')


X = train_df[X_features6]
y = train_df['win or loss']
X_predict = predict_df[X_features6]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#normalizing values
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_predict_scaled = scaler.transform(X_predict)

#print(week_to_predict.head(2)[['team_name', 'opponent name', 'total points scored szn avg', 'total points allowed szn avg',
#                               'opp total points scored szn avg', 'opp total points allowed szn avg']])
clf = LogisticRegression(max_iter=10000, penalty='l2', C=1)
clf.fit(X_train_scaled, y_train)


print("train accuracy score: ", accuracy_score(y_train, clf.predict(X_train_scaled)))
print("test accuracy score: ", accuracy_score(y_test, clf.predict(X_test_scaled)))
print("train roc score: ", roc_auc_score(y_train, clf.predict_proba(X_train_scaled)[:, 1]))
print("test roc score: ", roc_auc_score(y_test, clf.predict_proba(X_test_scaled)[:, 1]))



predictions = clf.predict(X_predict_scaled)
predictions_proba = clf.predict_proba(X_predict_scaled)

week_predictions = pd.DataFrame({'team': list(predict_df['team_name']), 'opponent':list(predict_df['opponent name']),
                                    'win': list(predictions),
                                    'win probability': list(predictions_proba)})
week_predictions['win probability'] = week_predictions['win probability'].apply(lambda x: x[1])
week_predictions['payoff needed'] = 1 / week_predictions['win probability']
save_path = week_folder + '\\week_predictions.csv'
#week_predictions.to_csv(save_path)
print(week_predictions)
