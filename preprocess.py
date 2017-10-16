import pandas as pd
import numpy as np
from config import num_feature_labels
from sklearn import preprocessing

# prepossess using sklearn module
gen_feature_data_training = pd.read_csv(
    './data/generated/gen_feature_data_training.csv').loc[:, 'DIFF_1':]
# feature scalar
scalar = preprocessing.StandardScaler()
feature_scalar = scalar.fit(gen_feature_data_training)


def fit_transform(data):
    return feature_scalar.fit_transform(
        data
    )


def gen_days_back(data, labels, days, starts_at=0):
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
    for k in range(starts_at, num_data):
        days_back_data = data[k - days:k]
        selected_day_back_data = days_back_data.loc[:, labels[0]:labels[-1]]
        selected_day_back_data_np = selected_day_back_data.values
        reshaped = np.reshape(selected_day_back_data_np.T,
                              (1, num_feature_labels))
        gen_data.append(reshaped)
    stacked = np.vstack(row for row in gen_data)
    data_frame = pd.DataFrame(data=stacked, columns=gen_labels)
    return data_frame, gen_labels
