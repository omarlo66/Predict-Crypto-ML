
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import time

# Configuration
CRYPTO_LIST = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'SOL-USD', 'ADA-USD', 'DOGE-USD', 'DOT-USD', 'MATIC-USD']
PERIOD = '1y'  # 1 year historical data
INTERVAL = '1d'  # Daily data
TEST_SIZE = 0.2  # 20% test data
RSI_PERIOD = 14  # RSI lookback period
PREDICTION_DAYS = 60  # Number of days to look back for prediction

# Email configuration
EMAIL_SENDER = 'Your Email'
EMAIL_PASSWORD = 'Your Password'  # Use app-specific password for Gmail
EMAIL_RECIPIENT = 'Reciepent Email or a list' #if a list of reciepents remember to add a loop for sending emails

def get_crypto_data(symbol):
    """Fetch historical data for a single cryptocurrency"""
    data = yf.download(symbol, period=PERIOD, interval=INTERVAL)
    return data

def calculate_rsi(data, period=RSI_PERIOD):
    """Calculate Relative Strength Index (RSI)"""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def prepare_features(data):
    """Prepare features including RSI and technical indicators"""
    df = data.copy()

    # Calculate RSI
    df['RSI'] = calculate_rsi(df)

    # Calculate moving averages
    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()

    # Calculate Bollinger Bands correctly
    rolling_std = df['Close'].rolling(window=21).std()
    df['Upper_Bollinger'] = df['MA_21'] #+ (2 * rolling_std)
    df['Lower_Bollinger'] = df['MA_21'] #- (2 * rolling_std)

    # Calculate MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change()

    # Drop NaN values
    df = df.dropna()

    return df

def create_dataset(df, days=PREDICTION_DAYS):
    """Create time-series dataset for prediction"""
    X = []
    y = []

    for i in range(days, len(df)):
        # Features: OHLCV + technical indicators for past 'days' days
        features = df.iloc[i-days:i][['Open', 'High', 'Low', 'Close', 'Volume',
                                    'RSI', 'MA_7', 'MA_21', 'Upper_Bollinger',
                                    'Lower_Bollinger', 'MACD', 'Signal_Line',
                                    'Daily_Return']].values
        # Target: Next day's close price
        target = df.iloc[i]['Close']

        X.append(features)
        y.append(target)

    return np.array(X), np.array(y)

def train_model(X_train, y_train):
    """Train Random Forest Regressor model"""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_gainers(crypto_list):
    """Predict next close price for all cryptocurrencies and return top gainers"""
    predictions = {}
    models = {}

    for symbol in crypto_list:
        try:
            print(f"\nProcessing {symbol}...")

            # Get data
            data = get_crypto_data(symbol)
            if data.empty:
                print(f"No data available for {symbol}")
                continue

            # Prepare features
            df = prepare_features(data)

            # Create dataset
            X, y = create_dataset(df)

            if len(X) == 0 or len(y) == 0:
                print(f"Not enough data to create dataset for {symbol}")
                continue

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, shuffle=False)

            # Normalize data
            scaler = MinMaxScaler()
            # Reshape 3D data to 2D for scaling
            X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
            X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

            X_train_scaled = scaler.fit_transform(X_train_reshaped)
            X_test_scaled = scaler.transform(X_test_reshaped)

            # Reshape back to 3D
            X_train = X_train_scaled.reshape(X_train.shape)
            X_test = X_test_scaled.reshape(X_test.shape)

            # Train model
            model = train_model(X_train.reshape(X_train.shape[0], -1), y_train)
            models[symbol] = (model, scaler)

            # Make prediction using most recent data
            recent_data = X[-1].reshape(1, -1)
            recent_data_scaled = scaler.transform(recent_data)
            predicted_price = model.predict(recent_data_scaled)[0]

            current_price = df['Close'].iloc[-1]
            percent_change = ((predicted_price - current_price) / current_price) * 100

            predictions[symbol] = {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'percent_change': percent_change
            }

            print(f"Prediction for {symbol}: Current ${current_price:.2f}, Predicted ${predicted_price:.2f} ({percent_change:.2f}%)")

        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue

    # Sort by predicted percentage gain
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1]['percent_change'], reverse=True)

    return sorted_predictions[:3], models  # Return top 3 gainers

def send_email(top_gainers):
    """Send email with top gainers information"""
    if not top_gainers:
        print("No predictions to send")
        return

    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECIPIENT
    msg['Subject'] = f"Crypto Top Gainers Prediction - {datetime.now().strftime('%Y-%m-%d')}"

    # Create email body
    body = "Top 3 Cryptocurrencies Predicted to Gain:\n\n"
    for i, (symbol, data) in enumerate(top_gainers, 1):
        body += f"{i}. {symbol}:\n"
        body += f"   Current Price: ${data['current_price']:.2f}\n"
        body += f"   Predicted Price: ${data['predicted_price']:.2f}\n"
        body += f"   Predicted Gain: {data['percent_change']:.2f}%\n\n"

    body += "\nThis is an automated prediction based on machine learning models. Please do your own research before making any investment decisions."

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

def main():
    """Main execution function"""
    print("Starting crypto prediction system...")

    while True:
        try:
            print(f"\nRunning prediction at {datetime.now()}")

            # Get top gainers predictions
            top_gainers, models = predict_gainers(CRYPTO_LIST)

            if not top_gainers:
                print("No valid predictions were made. Waiting before retrying...")
                time.sleep(60 * 60)  # Wait 1 hour before retrying
                continue

            # Print results
            print("\nTop 3 Predicted Gainers:")
            for i, (symbol, data) in enumerate(top_gainers, 1):
                print(f"{i}. {symbol}: Predicted gain of {data['percent_change']:.2f}%")

            # Send email
            send_email(top_gainers)

            # Sleep for 24 hours before next run
            print("\nWaiting for 24 hours before next prediction...")
            time.sleep(24 * 60 * 60)

        except KeyboardInterrupt:
            print("\nStopping prediction system...")
            break
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            print("Retrying in 1 hour...")
            time.sleep(60 * 60)

if __name__ == "__main__":
    main()