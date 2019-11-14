import numpy as np
import pandas as pd
from configparser import ConfigParser


cfg = ConfigParser()
cfg.read('../config.cfg')
DEFAULTS = cfg['DEFAULT']


def get_df_from_csv(filename):
    """
    Read in a csv file as a dataframe and clean the data
    :param filename:
    :param training:
    :return:
    """
    df = pd.read_csv(filename, na_values=DEFAULTS['MissingValues'])
    # df = clean_data(df, training)
    return df
