# Stock Price Predictor 📈

A simple stock price prediction model using Linear Regression for intern project work.

## Project Structure

```
Stock Prediction/
├── stock_predictor.py    # Main prediction script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Features

- Downloads real-time stock data from Yahoo Finance
- Uses Linear Regression for price prediction
- Calculates moving averages as features
- Visualizes actual vs predicted prices
- Predicts future stock prices

## How to Run

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Predictor

```bash
python stock_predictor.py
```

### Step 3: Enter a Stock Ticker

When prompted, enter a stock ticker symbol like:
- `AAPL` - Apple
- `GOOGL` - Google
- `MSFT` - Microsoft
- `TSLA` - Tesla
- `AMZN` - Amazon

## How It Works

1. **Data Collection**: Downloads 2 years of historical stock data
2. **Feature Engineering**: Creates features like:
   - Day number (time)
   - 5-day moving average
   - 10-day moving average
3. **Model Training**: Trains a Linear Regression model
4. **Evaluation**: Shows MSE, RMSE, and R² score
5. **Visualization**: Plots actual vs predicted prices
6. **Prediction**: Predicts the next 5 days

## Output

The program will:
- Print model evaluation metrics
- Save a plot as `prediction_results.png`
- Show predicted prices for the next 5 days

## Example Output

```
==================================================
    STOCK PRICE PREDICTOR
    Using Linear Regression
==================================================

Enter stock ticker: AAPL
Downloading data for AAPL...
Downloaded 504 days of data

MODEL EVALUATION
==================================================
Root Mean Squared Error (RMSE): $3.45
R² Score: 0.9234 (92.34%)

PREDICTING NEXT 5 DAYS
==================================================
Current Price: $185.50
Day +1: $186.20 (+0.38%)
Day +2: $186.90 (+0.75%)
...
```

## Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **scikit-learn** - Machine learning model
- **yfinance** - Stock data API
- **matplotlib** - Data visualization

## Notes

⚠️ **Disclaimer**: This is a simple educational project. Stock predictions are not financial advice. Real stock markets are influenced by many factors that simple models cannot capture.

## Author

Intern Project - Stock Price Predictor
