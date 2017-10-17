import numpy as np
import pandas as pd
from sklearn import preprocessing

from config import num_feature_labels

# prepossess using sklearn module
gen_feature_data_training = pd.read_csv(
    './data/generated/gen_feature_data_training.csv').loc[:, 'DIFF_1':]
# feature scalar
scalar = preprocessing.StandardScaler()
feature_scalar = scalar.fit(gen_feature_data_training)


# FIXME reverse the procedure of gen_days_back fn and fit_transform
# procedure right now gen_days_back -> fit_transform
# cause the gen_days_back fn generated lots of duplicated data
def fit_transform(data):
    return feature_scalar.fit_transform(
        data
    )


def gen_days_back(data, labels, days, starts_at=None):
    """
        generate data by days_before back
        using "days_before" back data to predict BTC_USD value
    """
    gen_labels = []
    gen_data = []
    num_data = data.shape[0]
    for label in labels:
        for j in range(days):
            gen_labels.append(label + '_' + str(j + 1))

    if starts_at == None:
        starts_at = days

    for k in range(starts_at - 1, num_data):
        _from = k - days + 1
        _to = k + 1
        days_back_data = data[_from:_to]
        selected_day_back_data = days_back_data.loc[:, labels[0]:labels[-1]]
        selected_day_back_data_np = selected_day_back_data.values
        reshaped = np.reshape(selected_day_back_data_np.T,
                              (1, num_feature_labels))
        gen_data.append(reshaped)
    stacked = np.vstack(row for row in gen_data)
    data_frame = pd.DataFrame(data=stacked, columns=gen_labels)
    return data_frame, gen_labels


def interpolate(data_frame):
    """
    Fn to deal with Null data
    :param data_frame: a pandas data frame object
    :return: interpolated data frame object
    """
    return data_frame.interpolate()
