import numpy as np
import pandas as pd
from configparser import ConfigParser
from sklearn.preprocessing import LabelEncoder


cfg = ConfigParser()
cfg.read('../config.cfg')
DEFAULTS = cfg['DEFAULT']


def get_trainings_dfs():
    """
    Read in the training data csv file and return a cleaned dataframe for each organization
    :return:
    """
    missing_values = DEFAULTS['MissingValues'].split(',')
    df = pd.read_csv(DEFAULTS['TrainingFile'], na_values=missing_values)

    jabref_df, myvolts_df, homepage_df = split_data(df)

    jabref_df = clean_jabref(jabref_df)
    myvolts_df = clean_myvolts(myvolts_df)
    homepage_df = clean_homepage(homepage_df)

    return jabref_df, myvolts_df, homepage_df


def get_test_dfs():
    """
    Read in the test data csv file and return a dataframe for each organization
    :return:
    """
    df = pd.read_csv(DEFAULTS['TestFile'], na_values=DEFAULTS['MissingValues'])

    jabref_df, myvolts_df, homepage_df = split_data(df)

    return jabref_df, myvolts_df, homepage_df


def split_data(df):
    """
    Split dataset into three subsets, one for each organization
    :param df:
    :return:
    """
    jabref_df = df[df['organization_id'] == int(DEFAULTS['JabRef'])]
    myvolts_df = df[df['organization_id'] == int(DEFAULTS['MyVolts'])]
    homepage_df = df[df['organization_id'] == int(DEFAULTS['HomePage'])]

    return jabref_df, myvolts_df, homepage_df


def clean_jabref(df):
    return df.drop(DEFAULTS['JabRefDroppedCols'].split(','), axis=1)


def clean_myvolts(df):
    dropped_cols = DEFAULTS['MyVoltsDroppedCols'].split(',') + DEFAULTS['MyVoltsIgnoredCols'].split(',')
    new_df = df.drop(dropped_cols, axis=1)

    for col in DEFAULTS['MyVoltsNumberCols'].split(','):
        mean = new_df[col].mean()
        new_df[col].fillna(mean, inplace=True)
    new_df.fillna('unknown', inplace=True)

    encode_cols = DEFAULTS['MyVoltsEncodeCols'].split(',')
    # new_df = oh_encode(new_df, encode_cols)
    new_df = label_encode(new_df, encode_cols)

    return new_df


def clean_homepage(df):
    return df.drop(DEFAULTS['HomePageDroppedCols'].split(','), axis=1)


def oh_encode(df, cols):
    """
    One Hot Encode the columns defined in DUMMY_COLS
    :param df:
    :return:
    """
    for col in cols:
        df = pd.concat((df.drop(columns=col), pd.get_dummies(df[col], drop_first=True)), axis=1)
    return df


def label_encode(df, cols):
    le = LabelEncoder()
    for col in cols:
        df[col] = le.fit_transform(df[col])
    return df
