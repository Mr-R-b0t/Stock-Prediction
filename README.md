# Stock Price Prediction with Multiple Models

## 

This project explores different approaches to predicting stock prices, using both traditional time series forecasting and deep learning techniques.

### File 1: `app.py`

**Purpose:**  A Streamlit web application for visualizing historical stock data and generating forecasts using the Prophet library.

**How it works:**

1. **Data Fetching:** Retrieves historical stock data from Yahoo Finance using `yfinance`.
2. **User Interface:** Provides a user-friendly interface to select a stock, choose the forecast period (in years), and display the raw data.
3. **Forecasting:** Employs the Prophet library, a powerful time series forecasting tool from Facebook, to build a prediction model.
4. **Visualization:** Presents the forecasted stock prices alongside the raw data in interactive plots.

**Key Points:**

- **Prophet:** Well-suited for beginners in time series forecasting due to its ease of use and robust performance.
- **Streamlit:** Enables the rapid creation of interactive data applications with minimal effort.

### File 2: `xlstm.py`

**Purpose:** Explores a cutting-edge approach using the xLSTMLM architecture (transformer-based LSTM) to predict the next day's closing stock price.

**How it works:**

1. **Data Acquisition:** Downloads historical stock data using `yfinance`.
2. **Preprocessing:** Scales the data to a normalized range for effective neural network training.
3. **Data Handling:** Utilizes PyTorch's Dataset and DataLoader classes for efficient time series data management.
4. **Model Configuration:** Configures the xLSTMLM model using a YAML-like format for flexibility.
5. **Training:** Trains the model to minimize the mean squared error (MSE) loss, a common metric for regression tasks.
6. **Model Persistence:** Saves the trained model for future use and predictions.
7. **Testing:** Includes a function to evaluate the model on new data and print the predicted price.

**Key Points:**

- **xLSTMLM:** A state-of-the-art architecture originally designed for language modeling, adapted for time series prediction due to its ability to capture long-range dependencies.
- **PyTorch:** A powerful deep learning framework chosen for its flexibility, dynamic computation graph, and GPU acceleration.
- **Complexity:** This script showcases a more advanced approach, highlighting the potential of xLSTMLM for improved prediction accuracy.

### File 3: `lstm.py`

**Purpose:** Implements a basic LSTM (Long Short-Term Memory) model for stock price prediction and evaluates its accuracy.

**How it works:**

1. **Data Gathering:** Collects stock data using `yfinance`.
2. **Data Preparation:** Preprocesses data by smoothing, scaling, and calculating exponential moving averages.
3. **Sequence Creation:** Generates sequences of past stock prices as input for the LSTM model.
4. **Model Building:** Constructs a Keras Sequential model with an LSTM layer and compiles it for training.
5. **Model Evaluation:** Trains and evaluates the model on test data, plotting the predictions.
6. **Accuracy Calculation:** Assesses the model's performance using a custom accuracy metric.

**Key Points:**

- **Basic LSTM:** Offers a simpler implementation compared to xLSTMLM for comparison and understanding of fundamentals.
- **Accuracy Metric:** Provides a way to quantify the model's prediction accuracy.

**Why xLSTMs are Better Than LSTMs**

- **Enhanced Long-Range Dependencies:** LSTMs are good at capturing short-term dependencies in sequential data. xLSTMs extend this capability by incorporating mechanisms (mLSTM and sLSTM blocks) that explicitly model long-range dependencies. This is crucial for financial time series where events far in the past can still influence current prices.
- **Scalability:** xLSTMs are designed with scalability in mind. They can handle much longer input sequences than traditional LSTMs due to their efficient attention mechanisms and sparse connections.
- **Adaptability:** The mLSTM and sLSTM blocks within xLSTMs are adaptable and can learn different types of patterns within the data, making them more versatile than standard LSTMs.
- **State-of-the-Art Performance:** xLSTMs have demonstrated state-of-the-art performance on various natural language processing tasks. Their ability to model long-range dependencies and scale to large inputs makes them promising for financial time series prediction as well.