from Portfolio_Class import Portfolio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

port2 = Portfolio(stocks=['AAPL', 'AMZN', 'NVDA', 'TSLA'], initial_capital=50000, weights=[0.25, 0.25, 0.25, 0.25], startDate="2016-01-08")

historical_returns = pd.Series(np.dot(port2.log_returns(), port2.weights)) # averaging log returns over the various stocks

days = 15
pv = port2.present_value()

range_returns = historical_returns.rolling(window = days).sum().dropna() # this gives an estimate for projected returns over 15 days

c_interval = 0.95
historical_VaR = -np.percentile(range_returns, 100 - (c_interval*100))*pv # looks at 5% of worst past returns
print(f'You can expect to lose ${round(historical_VaR,2)} over {days} days with {c_interval:.0%} confidence')

# Now I calculate the Expected Shortfall

projected_returns = pd.Series(range_returns)*pv
historical_losses = projected_returns[projected_returns >= historical_VaR]

historical_es = historical_losses.mean()
print(f'If you are guaranteed to lose more than ${round(historical_VaR,2)} then the Expected Shortfall over {days} days at the {c_interval:.0%} confidence level is ${round(historical_es,2)}')

# Let's see this visually!

plt.hist(projected_returns, bins=50, density=True)
plt.xlabel("Projected Gain/Loss ($)")
plt.ylabel("Frequency")
plt.title(f'Distribution of Portfolio Returns over {days} days')
plt.axvline(-historical_VaR, color='b', linestyle='dashed', linewidth=2, label=f'VaR at {c_interval:.0%} confidence level')
plt.axvline(-historical_es, color='r', linestyle='dashed', linewidth=2, label=f'Expected Shortfall at {c_interval:.0%} confidence level')
plt.legend()
plt.show()

# One of the big differences between this method and Monte Carlo, is that Monte Carlo is based on random simulations
# It is also advantageous against Parametric method as doesn't assume Normality
# but here there is nothing random! Purely determined from past data.