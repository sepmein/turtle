"""
    Generate training, cross validation and test data for the project
"""
import numpy as np
import pandas as pd

# read raw data
RAW_DATA = pd.read_csv('cleaned_data.csv')

# Define raw labels, from which function will generate feature labels
RAW_LABELS = [
    'DIFF', 'TRFEE', 'MKTCP', 'TOTBC', 'MWNUS', 'MWNTD', 'MWTRV', 'AVBLS',
    'BLCHS', 'ATRCT', 'MIREV', 'HRATE', 'CPTRA', 'CPTRV', 'TRVOU', 'TOUTV',
    'ETRVU', 'ETRAV', 'NTRBL', 'NADDU', 'NTREP', 'NTRAT', 'NTRAN'
]

# Define target label
TARGET_LABEL = ['MKPRU']

# Define machine learning ratio
TRAINING_RATIO = 0.6
CROSS_VALIDATION_RATIO = 0.3
TEST_RATIO = 0.1

# Define how much rows should be skipped
# Because at the initial year of bitcoin, there weren't any $-BTC data.
# So it should be skipped
STARTS_AT = 590

# Convert data string to pandas datatime format
RAW_DATA['Date'] = pd.to_datetime(RAW_DATA['Date'])



def gen_days_back(data, labels, days, starts_at):
    """
        generate data by days back
        using "days" back data to predict BTC_USD value
    """
    gen_labels = []
    gen_data = []
    for label in labels:
        for j in range(days):
            gen_labels.append(label + '_' + str(j + 1))
    for k in range(starts_at, data.shape[0]):
        days_back_data = data[k - days:k]
        selected_day_back_data = days_back_data.loc[:, 'DIFF':'NTRAN']
        selected_day_back_data_np = selected_day_back_data.values
        reshaped = np.reshape(selected_day_back_data_np.T,
                              (1, len(labels) * days))
        gen_data.append(reshaped)
    stacked = np.vstack(row for row in gen_data)
    dataframe = pd.DataFrame(data=stacked, columns=gen_labels)
    return dataframe, gen_labels

# Generate feature data
gen_feature_data, gen_labels = gen_days_back(
    data=RAW_DATA,
    labels=RAW_LABELS,
    days=50,
    starts_at=STARTS_AT
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
    label=TARGET_LABEL,
    starts_at=STARTS_AT
)

# Define random mask
MASK = np.random.rand(len(gen_feature_data))
MASK_TRAIN = MASK <= 0.6
MASK_CROSS_VALIDATION = (MASK > 0.6) & (MASK < 0.9)
MASK_TEST = MASK >= 0.9

# Generate three kinds of data
gen_feature_data_training = gen_feature_data[MASK_TRAIN]
gen_feature_data_cv = gen_feature_data[MASK_CROSS_VALIDATION]
gen_feature_data_test = gen_feature_data[MASK_TEST]
gen_target_data_training = target_data[MASK_TRAIN]
gen_target_data_cv = target_data[MASK_CROSS_VALIDATION]
gen_target_data_test = target_data[MASK_TEST]

# save three parts of data
gen_feature_data_training.to_csv('gen_feature_data_training.csv')
gen_target_data_training.to_csv('gen_target_data_training.csv')
gen_feature_data_cv.to_csv('gen_feature_data_cv.csv')
gen_target_data_cv.to_csv('gen_target_data_cv.csv')
gen_feature_data_test.to_csv('gen_feature_data_test.csv')
gen_target_data_test.to_csv('gen_target_data_test.csv')
