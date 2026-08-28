import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Portfolio_Class import Portfolio

#One thing I haven't done here is set up try and except commands when user is inputting Portfolio attributes
port1 = Portfolio(stocks= ['AAPL', 'AMZN', 'NVDA', 'MSFT'], initial_capital= 1000000, weights= [0.25,0.25,0.25,0.25], startDate= "2016-01-08")

def random_z_val():
    return np.random.normal(0,1)

days = 15
pv = port1.present_value()
er = port1.expected_return()
sd = port1.standard_deviation()

#Simulates the expected value of a portfolio given the standard deviation based on past returns over a specified time period
def simulated_gain_loss(z_val):
    return pv*er*days + pv*sd*z_val*np.sqrt(days)

#Run 15000 simulations

simulations = 15000
simReturn = []
for i in range(simulations):
    z_val = random_z_val()
    simReturn.append(simulated_gain_loss(z_val))

#Confidence Intervals and VaR calc
confidence_interval = 0.95
VaR = -np.percentile(simReturn,100*(1 - confidence_interval))
print(f'You can expect to lose ${round(VaR,2):,} over {days} days with {confidence_interval:.0%} confidence')



# What VaR doesn't tell us is the expected amount the portfolio can lose given we have gone past the VaR threshold.
# We can calculate the Conditional VaR or CVaR (or Expected Shortfall) that will tell us this preciscely.

# We have calculated the expected maximum loss given a certain confidence interval
# I can look at all the losses that are bigger than VaR, then take average of these.
# The average will be used as our estimate for the Expected Shortfall


sims = pd.Series(simReturn)
losses = sims[sims >= VaR] # this gives us a series of all our losses that where bigger that the VaR loss.

expected_shortfall = losses.mean() # effectively the same calc as historical but for the simulations
print(f'If you are guaranteed to lose more than ${round(VaR,2):,} then the Expected Shortfall over {days} days at the {confidence_interval:.0%} confidence level is ${round(expected_shortfall,2):,}')

# Plot of the simulations
plt.hist(simReturn, bins=50, density=True)
plt.xlabel("Simulated Gain/Loss ($)")
plt.ylabel("Frequency")
plt.title(f'Distribution of Portfolio Returns over {days} days')
plt.axvline(-VaR, color='b', linestyle='dashed', linewidth=2, label=f'VaR at {confidence_interval:.0%} confidence level')
plt.axvline(-expected_shortfall, color='r', linestyle='dashed', linewidth=2, label=f'Expected Shortfall at {confidence_interval:.0%} confidence level')
plt.legend()
plt.show()