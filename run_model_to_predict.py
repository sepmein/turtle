from model_loader import predict
from scrapper import scrap

feature = scrap()
prediction = predict(feature=feature)

print(prediction)
