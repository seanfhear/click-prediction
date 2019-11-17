import numpy as np
import pandas as pd
from configparser import ConfigParser
from src import preprocess

cfg = ConfigParser()
cfg.read('../config.cfg')
DEFAULTS = cfg['DEFAULT']


def get_train_and_test():
    """
    Returns a dataframe of the training dataset and a dataframe of the test dataset
    :return:
    """
    df_train = preprocess.get_trainings_dfs(DEFAULTS['TrainingDataFile'])
    df_test = preprocess.get_trainings_dfs(DEFAULTS['TestDataFile'])

    return df_train, df_test


def train(train):
    ''


def predict(train, test):
    ''


def main(train):
    """
    Runs train or predict based on the boolean passed in.
    :param train:
    :return:
    """
    train_data, test_data = get_train_and_test()
    if train:
        train(train_data)
    else:
        predict(train_data, test_data)


if __name__ == '__main__':
    main(train=1)
