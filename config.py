# Define raw labels, from which function will generate feature labels
feature_labels = ['DIFF', 'TRFEE', 'MKTCP', 'TOTBC', 'MWNUS',
                  'MWNTD', 'MWTRV', 'AVBLS', 'BLCHS', 'ATRCT',
                  'MIREV', 'HRATE', 'CPTRA', 'CPTRV', 'TRVOU',
                  'TOUTV', 'ETRVU', 'ETRAV', 'NTRBL', 'NADDU',
                  'NTREP', 'NTRAT', 'NTRAN', 'MKPRU']

# Define target label
target_label = ['MKPRU']

# Define machine learning ratio
training_set_split_ratio = 0.7
cross_validation_set_split_ratio = 0.2
test_set_split_ratio = 0.1

# Define how much rows should be skipped
# Because at the initial year of bitcoin, there weren't any $-BTC data.
# So it should be skipped
start_at = 500

# Learning rate
learning_rate = 0.001

# L2 norm lambda
lambd = 0.1
