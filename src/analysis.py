import pandas as pd
import matplotlib.pyplot as plt
from configparser import ConfigParser
from src import preprocess


cfg = ConfigParser()
cfg.read('../config.cfg', encoding='utf-8')
DEFAULTS = cfg['DEFAULT']


def count_values(df, col):
    print(pd.crosstab(df[col], df.set_clicked, normalize='index'))


def main():
    df_train = preprocess.get_df_from_csv(DEFAULTS['TrainingDataFile'])
    # count_values(df_train, 'recommendation_algorithm_id_used')
    count_values(df_train, 'organization_id')
    # pd.set_option('display.max_columns', 500)
    # print(df_train.describe())


if __name__ == '__main__':
    main()
