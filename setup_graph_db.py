import time
import yfinance as yf
import duckdb
import pandas as pd
print("Neo4j import successful!!!")


def fetchTickers():
    '''
    Fetches the list of NASDAQ-listed companies with the ticker symbol, company name, and more information.
        
    **Returns**: List of NASDAQ-listed companies with the ticker symbol and company name.
    '''
    # URLs for the NASDAQ Trader Symbol Directory files
    # List is updated everyday.
    nasdaq_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"

    # Load NASDAQ-listed stocks
    nasdaq_data = pd.read_csv(nasdaq_url, sep="|")
    # Remove the last row (footer info)
    nasdaq_data = nasdaq_data[:-1]

    #  Filter for companies that are not bankrupt, delisted, etc.
    normal_cmps = nasdaq_data.where(nasdaq_data["Financial Status"] == "N").dropna()
    normal_cmps.index = range(len(normal_cmps)) # type: ignore
    nasdaq_tickers = normal_cmps[["Symbol", "Security Name"]]
    nasdaq_tickers.loc[:, "Exchange"] = "NASDAQ"
    return nasdaq_tickers


if __name__ == "__main__":
    nasdaq_tickers = fetchTickers()
    print(list(nasdaq_tickers["Symbol"])) 