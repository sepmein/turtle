# Define raw labels, from which function will generate feature labels
feature_labels = ['DIFF', 'TRFEE', 'MKTCP', 'TOTBC', 'MWNUS',
                  'MWNTD', 'MWTRV', 'AVBLS', 'BLCHS', 'ATRCT',
                  'MIREV', 'HRATE', 'CPTRA', 'CPTRV', 'TRVOU',
                  'TOUTV', 'ETRVU', 'ETRAV', 'NTRBL', 'NADDU',
                  'NTREP', 'NTRAT', 'NTRAN', 'MKPRU']

# Define target label
target_label = ['MKPRU']

# How much days_before back
days_before = 50

# num of feature labels
num_feature_labels = len(feature_labels) * days_before

# Define machine learning ratio
training_set_split_ratio = 0.6
cross_validation_set_split_ratio = 0.3
test_set_split_ratio = 0.1

# Define how much rows should be skipped
# Because at the initial year of bitcoin, there weren't any $-BTC data.

start_at = 590

# Learning rate
learning_rate = 0.00001

# L2 norm lambda
lambd = 10


def generate_feature_label(labels, days):
    """
        generate data by days_before back
        using "days_before" back data to predict BTC_USD value
    """
    gen_labels = []
    for label in labels:
        for j in range(days):
            gen_labels.append(label + '_' + str(j + 1))
    return gen_labels


# Generate feature labels
generated_feature_labels = generate_feature_label(feature_labels, days_before)

# training steps
training_steps = 500000

# summary config
logdir = 'c:\\test_sum'

# record intervals
record_interval = 10
