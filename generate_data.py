"""
    Generate training, cross validation and test data for the project
"""
import numpy as np
import pandas as pd

from config import feature_labels, start_at, target_label, \
    training_set_split_ratio, cross_validation_set_split_ratio, num_feature_labels, days_before

# read raw data
RAW_DATA = pd.read_csv('./data/raw/data_071011.csv')

# Convert data string to pandas datatime format
RAW_DATA['Date'] = pd.to_datetime(RAW_DATA['Date'])


def gen_days_back(data, labels, days, starts_at):
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
        selected_day_back_data = days_back_data.loc[:, feature_labels[0]:feature_labels[-1]]
        selected_day_back_data_np = selected_day_back_data.values
        reshaped = np.reshape(selected_day_back_data_np.T,
                              (1, num_feature_labels))
        gen_data.append(reshaped)
    stacked = np.vstack(row for row in gen_data)
    data_frame = pd.DataFrame(data=stacked, columns=gen_labels)
    return data_frame, gen_labels


# Generate feature data
gen_feature_data, gen_labels = gen_days_back(
    data=RAW_DATA,
    labels=feature_labels,
    days=days_before,
    starts_at=start_at
)


# Generate target data fn
def gen_target_data(data, label, starts_at):
    """
        generate raw target data
    """
    result = data[starts_at:][label]
    return result


# Generate target data
target_data = gen_target_data(
    data=RAW_DATA,
    label=target_label,
    starts_at=start_at
)

# Define random mask
MASK = np.random.rand(len(gen_feature_data))
MASK_TRAIN = MASK <= training_set_split_ratio
MASK_CROSS_VALIDATION = (MASK > training_set_split_ratio) & \
                        (MASK < (training_set_split_ratio + cross_validation_set_split_ratio))
MASK_TEST = MASK >= (training_set_split_ratio + cross_validation_set_split_ratio)

# Generate three kinds of data
gen_feature_data_training = gen_feature_data[MASK_TRAIN]
gen_feature_data_cv = gen_feature_data[MASK_CROSS_VALIDATION]
gen_feature_data_test = gen_feature_data[MASK_TEST]
gen_target_data_training = target_data[MASK_TRAIN]
gen_target_data_cv = target_data[MASK_CROSS_VALIDATION]
gen_target_data_test = target_data[MASK_TEST]

# save three parts of data
gen_feature_data_training.to_csv('./data/generated/gen_feature_data_training.csv')
gen_target_data_training.to_csv('./data/generated/gen_target_data_training.csv')
gen_feature_data_cv.to_csv('./data/generated/gen_feature_data_cv.csv')
gen_target_data_cv.to_csv('./data/generated/gen_target_data_cv.csv')
gen_feature_data_test.to_csv('./data/generated/gen_feature_data_test.csv')
gen_target_data_test.to_csv('./data/generated/gen_target_data_test.csv')
