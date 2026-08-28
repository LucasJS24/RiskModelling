from Portfolio_Class import Portfolio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

port3 = Portfolio(stocks=['AAPL', 'MSFT', 'TSLA', 'NVDA'], initial_capital=100000, weights=[0.25, 0.25, 0.25, 0.25],
                  startDate='2016-01-08')
pm_pv = port3.present_value()

pm_returns = pd.Series(np.dot(port3.log_returns(), port3.weights))  # same as historical method

days = 15

pm_rolling_returns = pm_returns.rolling(window=days).sum().dropna()  # again same as historical method

pm_std_dev = port3.standard_deviation()

c_levels = [0.9, 0.95, 0.99]

VaRs = []
for cl in c_levels:
    VaRs.append(pm_pv * pm_std_dev * norm.ppf(cl) * np.sqrt(days) - pm_pv * pm_returns.mean() * days)

print(f'{"Confidence Level":<20} {"Value at Risk":<20}')
print('-' * 40)

# This prints each Confidence Level and it's corresponding VaR
for cl, VaR in zip(c_levels, VaRs):
    print(f'{cl * 100:>6.0f}%: {" ":<10} ${VaR:>10,.2f}')

# Expected Shortfall Calc
# es calc is different using parametric method to that of historical method

pm_expected_shortfall = []

for cl in c_levels:
    z = norm.ppf(cl)
    es = -pm_pv * pm_returns.mean() * days + pm_pv * pm_std_dev * np.sqrt(days) * norm.pdf(z) / (1-cl)
    pm_expected_shortfall.append(es)

print('\n')
print(f'{"Confidence Level":<20} {"Expected Shortfall":<20}')
print('-' * 40)

# This prints each Confidence Level and it's corresponding Expected Shortfall
for cl, es in zip(c_levels, pm_expected_shortfall):
    print(f'{cl * 100:>6.0f}%: {" ":<10} ${es:>10,.2f}')

# Let's visualise!
for cl, VaR, es in zip(c_levels, VaRs, pm_expected_shortfall):
    plt.hist(pm_pv * pm_rolling_returns, bins=150, density=True, alpha=0.5, label=f'{days}-Day Returns')
    plt.xlabel(f'{days}-Day Portfolio Returns ($)')
    plt.ylabel("Frequency")
    plt.title(f'Distribution of Portfolio Returns over {days} days ({cl:.0%} Level)')
    plt.axvline(x=-VaR, linestyle='--', color='b', label='VaR')
    plt.axvline(x=-es, linestyle='--', color='r', label='Expected Shortfall')
    plt.legend()
    plt.show()
