import numpy as np
import pandas as pd
RAW_DATA = pd.read_csv('cleaned_data.csv')
# Define machine learning ratio
TRAINING_RATIO = 0.6
CROSS_VALIDATION_RATIO = 0.3
TEST_RATIO = 0.1

# Generate three parts of data
MASK = np.random.rand(len(RAW_DATA))
MASK_TRAIN = MASK <= 0.6
MASK_CROSS_VALIDATION = (MASK > 0.6) & (MASK < 0.9)
MASK_TEST = MASK >= 0.9
RAW_DATA_TRAIN = RAW_DATA[MASK_TRAIN]
RAW_DATA_CROSS_VALIDATION = RAW_DATA[MASK_CROSS_VALIDATION]
RAW_DATA_TEST = RAW_DATA[MASK_TEST]
# save three parts of data
RAW_DATA_TRAIN.to_csv('raw_data_TRAIN.csv')
RAW_DATA_CROSS_VALIDATION.to_csv('raw_data_CROSS_VALIDATION.csv')
RAW_DATA_TEST.to_csv('raw_data_TEST.csv')
