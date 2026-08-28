#When dealing with several Portfolios each of similar structure it might be a wise idea to set up a class
#For example when I say structure I could mean that every portfolio has a list of assets, initial value invested and weightings to these assets
#this could be something I can incorporate into a class

import yfinance as yf
import numpy as np
import datetime as dt
import pandas as pd

class Portfolio:
    def __init__(self, stocks, initial_capital, weights, startDate):
        self.stocks = stocks
        self.initial_capital = initial_capital
        self.weights = pd.Series(weights, dtype=float, index=stocks)
        self.startDate = startDate
        self.close_prices = self.download_close_prices()

    def download_close_prices(self):
        return pd.DataFrame(yf.download(self.stocks, start=self.startDate, end=dt.datetime.now())['Close'])

    # based on the initial capital, weightings and stock prices this will tell us how many stocks is being held for each stock in the portfolio
    # I had trouble here keeping track of which items are pandas vs numpy. Also got slightly confused with the indexing at one point
    # I managed to resolve the issues though.
    def holdings(self):
        initial_prices = self.close_prices.iloc[0]
        return self.weights * self.initial_capital / initial_prices

    # tells us the value of the portfolio at each time step formatted as a pandas series, assuming the holdings don't change at each time step
    def port_values(self):
        prices = self.close_prices.to_numpy()
        hds = self.holdings().to_numpy()
        return pd.Series(prices @ hds)

    # net value of the portfolio at current day
    def present_value(self):
        pvs = self.port_values()
        return pvs.iloc[-1]

    # Note: The following methods are functions of the stock data itself, and not the portfolio.

    def returns(self):
        return (self.close_prices/self.close_prices.shift(1) - 1).dropna()

    #often better to work with log returns as they are easier computatinonally wise
    def log_returns(self):
        log_returns = np.log(self.close_prices/self.close_prices.shift(1))
        return log_returns[~np.isnan(log_returns).any(axis=1)]

    #better visualisation wtih the cumulative log returns rather than just log returns
    def cum_log_returns(self):
        return self.log_returns().cumsum()

    #Note here the expected return is based on the past data, not always the most accurate thing to do!
    def expected_return(self):
        return np.dot(self.weights, self.log_returns().mean())

    def standard_deviation(self):
        variance = self.weights.T @ self.cov_matrix().to_numpy() @ self.weights
        return np.sqrt(variance)

    def cov_matrix(self):
        return pd.DataFrame(self.log_returns()).cov()