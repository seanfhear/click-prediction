import numpy as np
import pandas as pd
from configparser import ConfigParser


cfg = ConfigParser()
cfg.read('../config.cfg')
DEFAULTS = cfg['DEFAULT']


def get_trainings_dfs(filename):
    """
    Read in a csv file and return a cleaned dataframe for each organization
    :param filename: training data filename
    :return:
    """
    df = pd.read_csv(filename, na_values=DEFAULTS['MissingValues'])

    jabref_df = clean_data(df, DEFAULTS['JabRef'])
    myvolts_df = clean_data(df, DEFAULTS['MyVolts'])
    homepage_df = clean_data(df, DEFAULTS['HomePage'])

    return jabref_df, myvolts_df, homepage_df


def clean_data(df, org):
    """
    Redirect to cleaning function for each of the three organizations in the dataset
    :param df:
    :param org: Integer. Which org the data is for, Jabref, MyVolts or JB's homepage
    :return:
    """
    if org == DEFAULTS['JabRef']:
        return clean_jabref(df)
    if org == DEFAULTS['MyVolts']:
        return clean_myvolts(df)
    if org == DEFAULTS['HomePage']:
        return clean_homepage(df)


def clean_jabref(df):
    new_df = df[df['organization_id'] == int(DEFAULTS['JabRef'])]\
        .drop(DEFAULTS['DroppedJabRefCols'].split(','), axis=1)

    return new_df


def clean_myvolts(df):
    new_df = df[df['organization_id'] == int(DEFAULTS['MyVolts'])]\
        .drop(DEFAULTS['DroppedMyVoltsCols'].split(','), axis=1)

    return new_df


def clean_homepage(df):
    new_df = df[df['organization_id'] == int(DEFAULTS['HomePage'])]\
        .drop(DEFAULTS['DroppedHomePageCols'].split(','), axis=1)

    return new_df
