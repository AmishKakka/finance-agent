import time
import yfinance as yf
import duckdb
import polars as pl
print("DuckDB import successful!!!")


data_points = ['symbol', 'shortName', 'currency', 'previousClose', 'open', 'dayLow', 'dayHigh', 'regularMarketPreviousClose',
               'regularMarketOpen', 'regularMarketDayLow', 'regularMarketDayHigh', 'dividendRate', 'dividendYield',
              'fiveYearAvgDividendYield', 'beta', 'trailingPE', 'forwardPE', 'volume',  'marketCap',
              'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'priceToSalesTrailing12Months', 'fiftyDayAverage', 
              'twoHundredDayAverage', 'profitMargins', 'bookValue', 'priceToBook', 'earningsQuarterlyGrowth', 
              'netIncomeToCommon', 'trailingEps', 'forwardEps', 'enterpriseToRevenue', 'enterpriseToEbitda', '52WeekChange',
              'ebitda', 'totalDebt', 'quickRatio', 'currentRatio', 'totalRevenue', 'debtToEquity', 'revenuePerShare', 'returnOnAssets', 
              'returnOnEquity', 'grossProfits', 'freeCashflow', 'operatingCashflow', 'earningsGrowth', 'revenueGrowth', 'grossMargins', 
              'ebitdaMargins', 'operatingMargins', 'trailingPegRatio']


def formatDatatoFile(data:list[list]):
    '''
    This function formats the data fetched from the API and saves it to a csv file.
    It does data cleaning and formatting.
    
    **Arguments**:
            data: Data fetched in list format from each company.
    '''
    df = pl.DataFrame(data)
    df = df.drop_nulls(subset=["symbol"])
    df = df.fill_null(0.0)
    df = df.rename({col: col.capitalize() for col in df.columns})

    # Scale and convert percentages using with_columns
    df = df.with_columns([
        (pl.col("Marketcap") / 1e8).alias("Marketcap"),           # in 100 millions
        (pl.col("Totalrevenue") / 1e8).alias("Totalrevenue"),
        (pl.col("Grossprofits") / 1e8).alias("Grossprofits"),
        (pl.col("Freecashflow") / 1e8).alias("Freecashflow"),
        (pl.col("Operatingcashflow") / 1e8).alias("Operatingcashflow"),

        (pl.col("Returnonequity") * 100).alias("Returnonequity"),   # in percentages
        (pl.col("Earningsgrowth") * 100).alias("Earningsgrowth"),
        (pl.col("Revenuegrowth") * 100).alias("Revenuegrowth"),
        (pl.col("Profitmargins") * 100).alias("Profitmargins"),
        (pl.col("Ebitdamargins") * 100).alias("Ebitdamargins"),
    ])

    # Filter out invalid currency rows
    df = df.filter(pl.col("Currency") != "0.0")
    # Round to 2 decimal places
    df = df.with_columns([pl.col(c).round(2) for c in df.columns if df.schema[c] in (pl.Float64, pl.Float32)])

    # Drop duplicates based on Symbol
    df = df.unique(subset=["Symbol"], keep="first")
    df.write_csv("stocksData.csv")
    print("Saved data to csv file.")

    duckdb.read_csv("stocksData.csv")
    print("Read data via Duck DB...")


def download_data(stocks):
    '''
        **Arguments**:
            stocks: List of stock tickers to download data for
    '''
    data = []
    stk = yf.Tickers(" ".join(stocks))
    for ticker in stocks:
        try:
            stock_info = stk.tickers[ticker].info
            row = {key: stock_info.get(key, None) for key in data_points}
            data.append(row)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    print(f"Done fetching data for {len(data)} stocks.")
    
    # The data we need :- 
    stocksData = [i for i in data if i!=None]
    formatDatatoFile(stocksData)


def fetchTickers():
    '''
    Fetches the list of NASDAQ-listed companies with the ticker symbol, company name, and more information.
        
    **Returns**: List of NASDAQ-listed companies with the ticker symbol and company name.
    '''
    # URLs for the NASDAQ Trader Symbol Directory files
    # List is updated everyday.
    nasdaq_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"

    # Load NASDAQ-listed stocks
    nasdaq_data = pl.read_csv(nasdaq_url, separator="|")
    # Remove the last row (footer info)
    nasdaq_data = nasdaq_data[:-1]

    #  Filter for companies that are not bankrupt, delisted, etc.
    normal = nasdaq_data.filter(pl.col("Financial Status") == "N")
    return normal.select(["Symbol", "Security Name"])


# ----------------------------------------------------------------------- #
if __name__ == "__main__":
    batch_size = 25
    nasdaq_tickers = fetchTickers()
    m = len(list(nasdaq_tickers["Symbol"]))

    for idx in range(0, m, batch_size):
        download_data(list(nasdaq_tickers["Symbol"][idx:idx+batch_size]))
        time.sleep(4) 