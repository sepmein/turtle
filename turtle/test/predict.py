from turtle.data_fetcher import preprocess
from turtle.data_fetcher import scrap
from turtle.model import predict

feature = scrap()
feature = preprocess.execute(feature)
prediction = predict(feature=feature)

print(prediction)
