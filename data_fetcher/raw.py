from datetime import datetime

from data_fetcher.scrapper import scrap_all

today = datetime.today()

data = scrap_all()
data.to_csv('raw.csv')
