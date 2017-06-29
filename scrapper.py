import quandl
quandl.ApiConfig.api_key = "6ywQ69kRqt26zAsHkFDP"
## data
DIFF = quandl.get("BCHAIN/DIFF")
TRFEE = quandl.get('BCHAIN/TRFEE')
MKTCP = quandl.get('BCHAIN/MKTCP')
TOTBC = quandl.get('BCHAIN/TOTBC')
MWNUS = quandl.get('BCHAIN/MWNUS')
BCDDY = quandl.get('BCHAIN/BCDDY')
BCDDM = quandl.get('BCHAIN/BCDDM')
BCDDE = quandl.get('BCHAIN/BCDDE')
TVTVR = quandl.get('BCHAIN/TVTVR')
NETDF = quandl.get('BCHAIN/NETDF')
MIOPM = quandl.get('BCHAIN/MIOPM')
MWNTD = quandl.get('BCHAIN/MWNTD')
MWTRV = quandl.get('BCHAIN/MWTRV')
AVBLS = quandl.get('BCHAIN/AVBLS')
BLCHS = quandl.get('BCHAIN/BLCHS')
ATRCT = quandl.get('BCHAIN/ATRCT')
MIREV = quandl.get('BCHAIN/MIREV')
HRATE = quandl.get('BCHAIN/HRATE')
CPTRA = quandl.get('BCHAIN/CPTRA')
CPTRV = quandl.get('BCHAIN/CPTRV')
TRVOU = quandl.get('BCHAIN/TRVOU')
TOUTV = quandl.get('BCHAIN/TOUTV')
ETRVU = quandl.get('BCHAIN/ETRVU')
ETRAV = quandl.get('BCHAIN/ETRAV')
NTRBL = quandl.get('BCHAIN/NTRBL')
NADDU = quandl.get('BCHAIN/NADDU')
NTREP = quandl.get('BCHAIN/NTREP')
NTRAT = quandl.get('BCHAIN/NTRAT')
NTRAN = quandl.get('BCHAIN/NTRAN')
## target value
MKPRU = quandl.get('BCHAIN/MKPRU')

## analysing the data
import numpy as np
import tensorflow as tf

