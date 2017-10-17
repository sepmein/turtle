"""
    Dnn Predictor
    This file serves to predict the value of BTC_US value
"""

# require packages
from data_fetcher.scrapper import scrap
from dnn_regressor import get_estimator, get_scaler, input_fn

# 1. Load the trained dnn_model
estimator = get_estimator()

# 2. Load the scaler
feature_scaler, target_scaler = get_scaler()

# 3. Load the data
data = scrap()

# 4. Scale the data by the scaler
# scaled_data = feature_scaler.fit_transform(data)

# 5. Build the input_fn for prediction
def input_fn_predict():
    features_tf = input_fn(feature=data)

# 6. Feed the predictor the data
result = estimator.predict(input_fn=input_fn_predict)

# 7. Inverse transform back the data
inverse_transformed_result = target_scaler.inverse_transform(result)

# 8. Return the data
print(inverse_transformed_result)
