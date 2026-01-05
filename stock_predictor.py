"""
Stock Price Predictor
A simple linear regression model to predict stock prices based on historical data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def download_stock_data(ticker, period="2y"):
    """
    Download historical stock data from Yahoo Finance.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'GOOGL')
        period: Time period for historical data
    
    Returns:
        DataFrame with stock data
    """
    print(f"Downloading data for {ticker}...")
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    print(f"Downloaded {len(data)} days of data")
    return data


def prepare_features(data):
    """
    Prepare features for the model.
    Uses simple features: day number, moving averages, price changes.
    
    Args:
        data: DataFrame with stock data
    
    Returns:
        X (features), y (target)
    """
    df = data.copy()
    
    # Create day number as primary feature
    df['Day'] = range(len(df))
    
    # Moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    
    # Price change from previous day
    df['Price_Change'] = df['Close'].pct_change()
    
    # Remove rows with NaN values
    df = df.dropna()
    
    # Features: Day, Moving Averages, Previous Close
    X = df[['Day', 'MA_5', 'MA_10']].values
    y = df['Close'].values
    
    return X, y, df


def train_model(X, y):
    """
    Train a Linear Regression model.
    
    Args:
        X: Feature matrix
        y: Target values
    
    Returns:
        Trained model, test data, predictions
    """
    # Split data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Don't shuffle to maintain time order
    )
    
    # Create and train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred


def evaluate_model(y_test, y_pred):
    """
    Evaluate model performance.
    
    Args:
        y_test: Actual values
        y_pred: Predicted values
    """
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    print(f"Mean Squared Error (MSE): ${mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"R² Score: {r2:.4f} ({r2*100:.2f}%)")
    print("="*50)
    
    return rmse, r2


def plot_results(y_test, y_pred, ticker):
    """
    Plot actual vs predicted prices.
    
    Args:
        y_test: Actual prices
        y_pred: Predicted prices
        ticker: Stock ticker symbol
    """
    plt.figure(figsize=(12, 6))
    
    # Plot actual vs predicted
    plt.subplot(1, 2, 1)
    plt.plot(y_test, label='Actual Price', color='blue', linewidth=2)
    plt.plot(y_pred, label='Predicted Price', color='red', linestyle='--', linewidth=2)
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred, alpha=0.5, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=150)
    plt.show()
    print("\nPlot saved as 'prediction_results.png'")


def predict_future(model, df, days=5):
    """
    Predict future stock prices.
    
    Args:
        model: Trained model
        df: DataFrame with historical data
        days: Number of days to predict
    """
    print(f"\n{'='*50}")
    print(f"PREDICTING NEXT {days} DAYS")
    print("="*50)
    
    last_day = df['Day'].iloc[-1]
    last_ma5 = df['MA_5'].iloc[-1]
    last_ma10 = df['MA_10'].iloc[-1]
    current_price = df['Close'].iloc[-1]
    
    print(f"Current Price: ${current_price:.2f}")
    print(f"\nFuture Predictions:")
    print("-"*30)
    
    previous_price = current_price  # Track previous day's price for day-over-day change
    
    for i in range(1, days + 1):
        # Simple prediction using last known moving averages
        future_features = np.array([[last_day + i, last_ma5, last_ma10]])
        predicted_price = model.predict(future_features)[0]
        change_from_current = ((predicted_price - current_price) / current_price) * 100
        change_from_previous = ((predicted_price - previous_price) / previous_price) * 100
        
        print(f"Day +{i}: ${predicted_price:.2f} (from start: {change_from_current:+.2f}%, daily: {change_from_previous:+.2f}%)")
        previous_price = predicted_price  # Update for next iteration


def main():
    """Main function to run the stock predictor."""
    print("="*50)
    print("    STOCK PRICE PREDICTOR")
    print("    Using Linear Regression")
    print("="*50)
    
    # Get stock ticker from user
    ticker = input("\nEnter stock ticker (e.g., AAPL, GOOGL, MSFT): ").upper().strip()
    
    if not ticker:
        ticker = "AAPL"  # Default to Apple
        print(f"Using default ticker: {ticker}")
    
    try:
        # Step 1: Download data
        data = download_stock_data(ticker)
        
        if len(data) < 30:
            print("Error: Not enough data. Please try a different ticker.")
            return
        
        # Step 2: Prepare features
        X, y, df = prepare_features(data)
        print(f"\nPrepared {len(X)} samples for training")
        
        # Step 3: Train model
        model, X_test, y_test, y_pred = train_model(X, y)
        print("Model trained successfully!")
        
        # Step 4: Evaluate model
        rmse, r2 = evaluate_model(y_test, y_pred)
        
        # Step 5: Plot results
        plot_results(y_test, y_pred, ticker)
        
        # Step 6: Predict future prices
        predict_future(model, df, days=5)
        
        print("\n" + "="*50)
        print("PREDICTION COMPLETE!")
        print("="*50)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Please check the ticker symbol and try again.")


if __name__ == "__main__":
    main()
