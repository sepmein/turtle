from datetime import datetime

from turtle.data_fetcher import scrap_all

today = datetime.today()

data = scrap_all()
data.to_csv('raw.csv')
