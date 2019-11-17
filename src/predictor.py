import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from configparser import ConfigParser
from datetime import datetime
from src import preprocess

cfg = ConfigParser()
cfg.read('../config.cfg')
DEFAULTS = cfg['DEFAULT']


def train(jabref_train, myvolts_train, homepage_train):
    start_time = datetime.now()
    myvolts_test = myvolts_train[DEFAULTS['Target']]
    myvolts_train = myvolts_train.drop(DEFAULTS['TrainingTargets'].split(','), axis=1)
    myvolts_X_train, myvolts_X_test, myvolts_y_train, myvolts_y_test = \
        train_test_split(myvolts_train, myvolts_test, test_size=0.3, random_state=109)

    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(myvolts_X_train, myvolts_y_train)
    y_pred = clf.predict(myvolts_X_test)

    df = pd.DataFrame({'Actual': myvolts_y_test, 'Predicted': y_pred.flatten()})
    df.to_csv(DEFAULTS['TrainingOutputFile'])

    print("Accuracy: ", metrics.accuracy_score(myvolts_y_test, y_pred))
    print("Precision: ", metrics.precision_score(myvolts_y_test, y_pred))
    print("Recall: ", metrics.recall_score(myvolts_y_test, y_pred))
    print("Time Taken: ", datetime.now() - start_time)


def predict(train, test):
    ''


def main(training):
    """
    Runs train or predict based on the boolean passed in.
    :param training:
    :return:
    """
    jabref_train, myvolts_train, homepage_train = preprocess.get_trainings_dfs()
    if training:
        train(jabref_train, myvolts_train, homepage_train)
    else:
        ''
        # predict(train_data, test_data)


if __name__ == '__main__':
    main(training=1)
