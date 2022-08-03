import pandas as pd
import yfinance
file = open("ticker.txt", "r+")
tickers = file.read().split("\n")
file.close()
database_dict = {}
i = 0
for ticker_id in tickers:
    ticker = yfinance.Ticker(ticker_id)
    data = ticker.history(period="max")["High"]
    data = data.tolist()
    for x in range(100, 100 + (len(data)//100) * 100, 100):
        database_dict[i] = data[x-100: x]
        i+=1

df = pd.DataFrame.from_dict(database_dict, orient="index")
df.to_csv("data.csv")