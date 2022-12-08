import numpy as np
import pandas as pd
import time
from numpy import *


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def get_pfr_data(year):
    team_name_dict = {"buffalo bills": 'buf',
             "new york jets": "nyj",
             "miami dolphins": 'mia',
             "new england patriots": "nwe",
             "baltimore ravens": 'rav',
             "cincinnati bengals": 'cin',
             "cleveland browns": 'cle',
             'pittsburgh steelers': 'pit',
             'tennessee titans': 'oti',
             'indianapolis colts': 'clt',
             "jacksonville jaguars": 'jax',
             'houston texans': 'htx',
             'kansas city chiefs': 'kan',
             'los angeles chargers': 'sdg',
             'denver broncos': 'den',
             'las vegas raiders': 'rai',
             'philadelphia eagles': 'phi',
             'dallas cowboys': 'dal',
             'new york giants': 'nyg',
             'washington commanders': 'was',
             'minnesota vikings': 'min',
             'greenbay packers': 'gnb',
             'chicago bears': 'chi',
             'detroit lions': 'det',
             'atlanta falcons': 'atl',
             'tampa bay buccaneers': 'tam',
             'new orleans saints': 'nor',
             'carolina panthers': 'car',
             'seattle seahawks': 'sea',
             'san francisco 49ers': 'sfo',
             'los angeles rams': 'ram',
             'arizona cardinals': 'crd'
             }
    # create url function
    def make_url(team_name, year):
        url = 'https://www.pro-football-reference.com/teams/' + str(team_name) + '/' + str(year) + '.htm'
        return url

    team_names = list(team_name_dict.values())
    pfr_data = []
    # for loop to iterate through all teams and extract data
    for key, team_name in team_name_dict.items():
        # print(team_name)

        # creating url to extract data from
        url = make_url(team_name, year)
        print(url)
        # extract out team season table, which is the second to last table on the page.
        team_data = pd.read_html(url)[-2]

        # dataframe comes with a multilevel index. First level is nothing, so we're dropping it
        team_data = team_data.droplevel(0, axis=1)

        # creating team name column
        team_data['team_name'] = key

        # adding season year column
        team_data['season'] = year


        pfr_data.append(team_data)

        # Telling the function to go to sleep for 2 seconds before sending another request to website
        # Too many requests too quickly results in a 'HTTP error 429 (Too Many Requests)''
        time.sleep(2)
    return pfr_data

def clean_pfr(df, week_to_predict):
    team_data = df
    team_data.drop(['Date', 'Day', 'OT', 'Unnamed: 3_level_1', 'Unnamed: 4_level_1', 'Offense', 'Defense',
                    'Sp. Tms'], inplace=True, axis=1)
    # renaming remaining columns into more readable names.
    col_renaming = ['week', 'win or loss', 'season record', 'home or away', 'opponent name',
                    'total points scored', 'total points allowed', 'first downs gained',
                    'total yards gained', 'pass yards gained', 'rush yards gained', 'turnovers lost',
                    'first downs allowed', 'total yards against', 'pass yards against',
                    'rush yards against', 'turnovers gained by defense', 'team_name', 'season']
    team_data.columns = col_renaming

    # making opponent names lowercase
    team_data['opponent name'] = team_data['opponent name'].str.lower()

    # Only keeping data up until the week of interest.
    team_data = team_data.iloc[0:week_to_predict]

    # extract out prediction week so it doesn't get dropped when we remove bye weeks as there will be null values for
    # latest season.
    prediction_data = pd.DataFrame(team_data[team_data['week'] == week_to_predict])

    team_data = team_data[team_data['week'] != week_to_predict]

    # removing bye weeks which have no data
    team_data = team_data[~team_data['win or loss'].isnull()]

    # Adding prediction week back to dataframe
    team_data = pd.concat([team_data, prediction_data])

    # convert home or away column to a dummy variable. 0 for home, 1 for away.
    team_data['home or away'].replace("\@", "1", inplace=True, regex=True)
    team_data['home or away'].replace(np.nan, "0", inplace=True)
    team_data['home or away'].replace("[a-zA-Z]", "1", regex=True, inplace=True)
    team_data['home or away'] = team_data['home or away'].astype('float64')

    # removing all ties as these happen so infrequently since the introduction of OT
    team_data = team_data[team_data['win or loss'] != 'T']

    # convert win or loss into numeric. 0 for loss, 1 for a win
    team_data['win or loss'].replace("W", 1, regex=True, inplace=True)
    team_data['win or loss'].replace("L", 0, regex=True, inplace=True)

    # converting target values into a float
    team_data['win or loss'] = team_data['win or loss'].astype('float64')

    # Any previous season's data (i.e. not 2022) may have some string values in the week column (i.e. 'playoff')
    # so we need to make sure that the week column is not still an obj and convert it to a float value.
    team_data['week'] = team_data['week'].astype('float64')

    team_data = team_data[team_data['opponent name'] != 'bye week']


    # these two columns have nan's where they should have 0's
    team_data['turnovers lost'].replace(np.nan, 0, inplace=True)
    team_data['turnovers gained by defense'].replace(np.nan, 0, inplace=True)

    #updating two team names to match our dictionary
    teams_to_rename = {"green bay packers": "greenbay packers", "washington football team": "washington commanders",
                       "washington redskins": "washington commanders", "oakland raiders": "las vegas raiders"}
    team_data.replace(teams_to_rename, regex=True, inplace=True)

    # only keep up until week of interest
    team_data = team_data[team_data['week'] <= week_to_predict]

    return team_data



def create_feature_cols(team_data_list):

    def get_season_avg(df, column):
        weeks_played = len(df)
        avg_szn = [np.nan]
        for i in range(weeks_played):
            if i == 0:
                continue
            else:
                weekly_szn_avg = df[0:i][column].mean()
                avg_szn.append(weekly_szn_avg)

        return avg_szn


    for _, team_data in enumerate(team_data_list):
        #creating win ratio feature by splitting the column "season record" into individual
        # win and loss columns
        team_data['wins'] = team_data['season record'].str.split('-').str[0].astype('float64')
        team_data['losses'] = team_data['season record'].str.split('-').str[1].astype('float64')
        team_data['season record'] = team_data['season record'].astype('str')
        team_data['OT losses'] = team_data['season record'].apply(lambda cell:
                                                                  cell.split('-')[-1] if len(cell.split('-'))>2
                                                                  else 0).astype('float64')
        team_data['win ratio'] = team_data['wins'] / (team_data['wins'] + team_data['losses'] + team_data['OT losses'])

        #update win/loss ratio to represent win losses heading into current game.
        #the data includes the win or loss data from current week which we will not know ahead of time.
        record = [np.nan]
        for wr in team_data['win ratio']:
            record.append(wr)
        team_data['win ratio'] = record[0:-1]

        #creating season long averages to be used as features
        #
        cols_to_create = ['total points scored', 'total points allowed', 'first downs gained',
                                  'total yards gained', 'pass yards gained', 'rush yards gained',
                                  'turnovers lost', 'first downs allowed', 'total yards against',
                                  'pass yards against', 'rush yards against',
                                  'turnovers gained by defense']
        for col in cols_to_create:
            new_col = col + " szn avg"
            team_data[new_col] = get_season_avg(team_data, col)

    return team_data_list


def get_opponent_stats(df):
    x = df.copy()
    x = x[['team_name', 'week', 'win ratio', 'total points scored szn avg',
          'total points allowed szn avg', 'first downs gained szn avg',
          'total yards gained szn avg', 'pass yards gained szn avg',
          'rush yards gained szn avg', 'turnovers lost szn avg',
          'first downs allowed szn avg', 'total yards against szn avg',
          'pass yards against szn avg', 'rush yards against szn avg',
          'turnovers gained by defense szn avg']]
    cols_to_rename = ['win ratio', 'total points scored szn avg',
                      'total points allowed szn avg', 'first downs gained szn avg',
                      'total yards gained szn avg', 'pass yards gained szn avg',
                      'rush yards gained szn avg', 'turnovers lost szn avg',
                      'first downs allowed szn avg', 'total yards against szn avg',
                      'pass yards against szn avg', 'rush yards against szn avg',
                      'turnovers gained by defense szn avg']

    col_renaming = [" ".join(('opp', col)) for col in cols_to_rename]
    print(col_renaming)
    col_renaming = dict(zip(cols_to_rename, col_renaming))
    print(col_renaming)
    x.rename(col_renaming, inplace=True, axis=1)
    x.rename({'team_name':'opponent name'}, inplace=True, axis=1)
    df = df.merge(x, how='left', on=['opponent name', 'week'])
    '''
    col_to_return = []
    opponent_name = list(df['opponent name'])
    week = list(df['week'])

    print(df[(df['team_name']=='buffalo bills') & (df['week']==2)]['total points scored szn avg'].values[0])
    #print(df[(df['opponent name']==opponent_name[0]) & (df['week']==week[0])][col])
    for i in range(len(opponent_name)):
        print(opponent_name[i])
        print(week[i])
        print(df[(df['team_name']==opponent_name[i]) & (df['week']==week[i])])
        # print('opponent name: ', opponent_name[i], 'week: ', week[i])
        col_value = df[(df['team_name']==opponent_name[i]) & (df['week']==week[i])][col].values[0]
        col_to_return.append(col_value)'''
    return df

def get_clean_data(year, week):
    week_folder = r"C:\Users\lmoor\Desktop\Data Science Projects\NFL Predictions\week " + str(week) + '\df_' + str(year) + '.csv'
    x = get_pfr_data(year)
    for i, df in enumerate(x):
        x[i] = clean_pfr(df, week)
    x = create_feature_cols(x)
    x = pd.concat(x)

    x = get_opponent_stats(x)

    x.to_csv(week_folder)




