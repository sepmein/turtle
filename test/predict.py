from data_fetcher import preprocess
from data_fetcher.scrapper import scrap
from model.estimator import predict

feature = scrap()
feature = preprocess.execute(feature)
prediction = predict(feature=feature)

print(prediction)
