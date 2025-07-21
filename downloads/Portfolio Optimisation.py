import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

tickers = ["AZN", "SHEL.L", "HSBA.L", "ULVR.L", "REL.L"]
for ticker in tickers:
    print(ticker)

    # Fetch historical data
    data = yf.download(ticker, start="2018-01-01", end="2023-01-01")

    # Display the first few rows of the dataframe
    print(data.head())

    # Normalize the closing prices
    data['Normalized'] = data['Close'] / data['Close'].iloc[0] * 100

    # Calculating simple moving averages (SMA)
    data['SMA_1'] = data['Close'].rolling(window=1).mean()
    data['SMA_7'] = data['Close'].rolling(window=7).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_25'] = data['Close'].rolling(window=25).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_150'] = data['Close'].rolling(window=150).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # Calculate daily returns
    data['Daily_Return'] = data['Close'].pct_change()

    # Combine 'Close' and 'Volume' into one dataset
    close_volume_data = data[['Close', 'Volume']]

    # Get description of the combined dataset
    print("\nCombined Close and Volume Description:")
    print(close_volume_data.describe())

    # Calculate and print the median
    median_close = data['Close'].median()
    print(f"\nMedian: {median_close}")
    
    # Calculate annualized mean return & SD
    annualized_mean_return = data['Daily_Return'].mean() * 252
    annualized_std_dev = data['Daily_Return'].std() * np.sqrt(252)

    cumulative_returns = (1 + data['Daily_Return']).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    print(f"Annualized Mean Return: {annualized_mean_return}")
    print(f"Annualized Volatility: {annualized_std_dev}")
    print(f"Max Drawdown: {max_drawdown}")

    # Display the updated dataframe
    print(data[['Normalized', 'Daily_Return', 'SMA_7']].head(10))

    # Plotting the volume
    plt.figure(figsize=(12, 6))
    plt.plot(data['Volume'], label='Volume', color='green')

    # Overwrite last plot
    fig1 = plt.figure(1)

    plt.title(ticker+' Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.show()

    # Calculate the 20-period standard deviation (SD)
    data['SD'] = data['Close'].rolling(window=30).std()

    # Calculate the Upper bollinger band (UB) and lower bollinger Band (LB)
    data['UB'] = data['SMA_20'] + 2 * data['SD']
    data['LB'] = data['SMA_20'] - 2 * data['SD']

    # Identify buy/sell signals
    data['Bollinger_Signal_UB'] = np.where(data['SMA_1'] > data['UB'], 1, 0)
    data['Bollinger_Position_UB'] = data['Bollinger_Signal_UB'].diff()
    data['Bollinger_Signal_LB'] = np.where(data['SMA_1'] < data['LB'], 1, 0)
    data['Bollinger_Position_LB'] = data['Bollinger_Signal_LB'].diff()  

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot the price chart
    plt.plot(data.index, data['SMA_1'], label='Close Price', color='black')

    # Plot the upper bollinger band (UB)
    plt.plot(data.index, data['UB'], label='Upper Bollinger Band', color='red')
    plt.plot(data.index, data['LB'], label='Lower Bollinger Band', color='green')
    plt.fill_between(data.index, data['LB'], data['UB'], color='gray', alpha=0.3, label='Bollinger Band Area')

    # Plot the middle bollinger band (SMA)
    plt.plot(data.index, data['SMA_20'], label='Middle Bollinger Band', color='blue')

    # Plot the buy/sell signals
    plt.plot(data[data['Bollinger_Position_LB'] == 1].index, data['Close'][data['Bollinger_Position_LB'] == 1], '^', markersize=10, color='green', label='Buy Signal')
    plt.plot(data[data['Bollinger_Position_UB'] == 1].index, data['Close'][data['Bollinger_Position_UB'] == 1], 'v', markersize=10, color='red', label='Sell Signal')

    # Customize the chart layout
    plt.title(ticker+' Stock Price with Bollinger Bands Strategy')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Generate buy/sell signals based on SMA crossover
    data['SMA_Signal'] = np.where(data['SMA_50'] > data['SMA_200'], 1, 0)
    data['SMA_Position'] = data['SMA_Signal'].diff()  # Identify buy/sell transitions

    # Plot the SMA crossover strategy with buy/sell signals
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close Price', alpha=0.6)
    plt.plot(data['SMA_50'], label='50-Day SMA', color='blue')
    plt.plot(data['SMA_200'], label='200-Day SMA', color='orange')
    plt.plot(data[data['SMA_Position'] == 1].index, data['Close'][data['SMA_Position'] == 1], '^', markersize=10, color='green', label='Buy Signal')
    plt.plot(data[data['SMA_Position'] == -1].index, data['Close'][data['SMA_Position'] == -1], 'v', markersize=10, color='red', label='Sell Signal')

    # Overwrite last plot
    fig1 = plt.figure(1)

    # Customize the chart layout
    plt.title(ticker+' Stock Price with SMA Crossover Strategy')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Ensure extracted the 'Close' prices for the current ticker
    close_prices = data['Close'][ticker]  # Extract Series for this ticker

    # Debugging type
    # print(f"Close Prices Type: {type(close_prices)}")

    # Calculate daily percentage change for the 'Close' prices
    daily_returns = close_prices.pct_change()

    # Ensure signals are shifted correctly
    sma_signal = data['SMA_Signal'].shift(1)
    bollinger_signal_combined = data['Bollinger_Signal_UB'].shift(1) + data['Bollinger_Signal_LB'].shift(1)

    # Debugging signals
    # print(f"SMA Signal Type: {type(sma_signal)}")
    # print(f"Bollinger Signal Combined Type: {type(bollinger_signal_combined)}")

    # Calculate strategy returns
    data['SMA_Strategy_Returns'] = daily_returns * sma_signal
    data['Bollinger_Strategy_Returns'] = daily_returns * bollinger_signal_combined

    # Drop NaN values resulting from shifts
    data = data.dropna()

    # Calculate cumulative returns
    data['Cumulative_SMA_Returns'] = (1 + data['SMA_Strategy_Returns']).cumprod()
    data['Cumulative_Bollinger_Returns'] = (1 + data['Bollinger_Strategy_Returns']).cumprod()

    # Plot the cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(data['Cumulative_SMA_Returns'], label='SMA Strategy Cumulative Returns', color='blue')
    plt.plot(data['Cumulative_Bollinger_Returns'], label='Bollinger Bands Strategy Cumulative Returns', color='orange')

    # Overwrite last plot
    fig1 = plt.figure(1)

    # Customise the graph
    plt.title(f'Cumulative Returns Comparison: {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Fetching historical data
data = yf.download(tickers, start="2018-01-01", end="2023-01-01")['Close']

# Normalize the stock prices (starting at 100)
normalized_data = data / data.iloc[0] * 100

# Plot the normalized stock prices
plt.figure(figsize=(12, 6))

for ticker in tickers:
    plt.plot(normalized_data.index, normalized_data[ticker], label=ticker)

# Customize the graph
plt.title("Normalized Stock Prices of All Companies (2018-2023)")
plt.xlabel("Date")
plt.ylabel("Normalized Price (Starting at 100)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the graph
plt.show()

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate daily returns for the portfolio
portfolio_returns = data.pct_change().dropna()

# Risk-free rate for the Sharpe ratio calculation (4.75% based on UK official bank rate)
risk_free_rate = 0.0475 

# Define weights for the stocks
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

# Calculate portfolio return and standard deviation
portfolio_return = np.sum(portfolio_returns.mean() * weights) * 252
portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(portfolio_returns.cov() * 252, weights)))

# Show values
print("Optimal Portfolio:")
for ticker, weight in zip(tickers, weights):
    print(f"{ticker}: {weight:.2%}")
print(f"Expected Return: {portfolio_return}")
print(f"Standard Deviation: {portfolio_std_dev}")

# Calculate sharpe ratio
sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev

# Show value
print(f"Sharpe Ratio: {sharpe_ratio}")

# Expected returns and covariance matrix
mean_returns = returns.mean() * 252  # Annualize the mean returns
cov_matrix = returns.cov() * 252  # Annualize the covariance matrix

# Number of portfolios to simulate
num_portfolios = 10000

# Initialize variables to track the optimal portfolio
max_sharpe_ratio = -np.inf
optimal_weights = None

for i in range(num_portfolios):
    # Generate random portfolio weights
    weights = np.random.random(5)
    weights /= np.sum(weights)
    
    # Calculate portfolio return
    portfolio_return = np.sum(weights * mean_returns)
    
    # Calculate portfolio volatility
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Calculate sharpe ratio
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    # Check if this portfolio has a higher Sharpe ratio
    if sharpe_ratio > max_sharpe_ratio:
        max_sharpe_ratio = sharpe_ratio
        optimal_weights = weights

# Print the optimal portfolio
print("Optimal Portfolio:")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.2%}")

# Calculate optimal return and volatility
optimal_return = np.sum(optimal_weights * mean_returns)
optimal_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))

# Print expected returns, volatility, and sharpe ratio
print(f"Expected Return: {optimal_return}")
print(f"Volatility: {optimal_volatility}")
print(f"Sharpe Ratio: {max_sharpe_ratio}")
