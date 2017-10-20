from data_fetcher.input_fn import gen_feature_data_cv
from model.estimator import predict

result = predict(gen_feature_data_cv)
print(result)
