"""
    Scrap quandl for the bitcoin data
"""

# imports
from datetime import date, datetime, timedelta

import pandas as pd
import quandl

from turtle import api_key, days_before, all_labels

quandl.ApiConfig.api_key = api_key

all_labels = all_labels


def _scrap_(labels, start_date=None, end_date=None):
    """
    Internal scrap function
    :param labels:
    :param start_date:
    :param end_date:
    :return:
    """
    result = []
    for label in labels:
        temp = quandl.get('BCHAIN/' + label, start_date=start_date, end_date=end_date)
        result.append(temp)

    data_frame = pd.concat(result, axis=1)
    data_frame.columns = labels
    return data_frame


def scrap_all():
    """
    Get all raw data
    :return:
    """
    return _scrap_(all_labels)


def scrap(year=None, month=None, day=None):
    """
    To scrap bitcoin data from quandl and preprocess it using predefined preprocess module
    :param year: int represent year
    :param month: int represent month
    :param day: int represent day
    :return: a data frame object to be feed to model
    """
    # check out the date inputted is 'today' or not
    # FIXME quandl api timezone and china timezone not compatible, the latest data will be NaN ocassionally (8/24?)
    if year is None:
        target_date = datetime.today()
    else:
        target_date = date(year, month, day)

    # days back
    end_date = target_date - timedelta(days=1)
    days_interval = timedelta(days=days_before - 1)
    start_date = end_date - days_interval

    # get data
    data_frame = _scrap_(all_labels, start_date, end_date)

    return data_frame
