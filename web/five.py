import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf

# Function to split sequence
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Function to train and predict with SARIMA
def train_and_predict_sarima(X_train, y_train, X_test_copy, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    sarima_model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)
    sarima_fit = sarima_model.fit(disp=False)
    y_pred_sarima = sarima_fit.get_forecast(steps=len(X_test_copy)).predicted_mean
    residuals = sarima_fit.resid
    return y_pred_sarima, residuals

# Read data
df = pd.read_csv("C:\\Users\\harsh\\OneDrive\Desktop\\train_21.csv")
df = df.drop('Page', axis=1)

# define input sequence
row = df.iloc[105, :].values
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(row, n_steps)

# Reshape for SARIMA input shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Streamlit app UI code starts here
st.set_page_config(
    page_title="Chrona.ai: A Web Traffic Prediction App",
    layout="wide"
)

st.markdown("<h1 class='centered-title'> 🚀 Chrona.ai 🚀 </h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Add a brief passage about your app, including SARIMA and ACF details
st.markdown(
    """
    Welcome to Chrona.ai, your favorite Web Traffic Time Series Prediction Web App!
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.sidebar.title('Select an option')
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Display styled text on the sidebar
st.sidebar.markdown(
    """
    <div style="width: 200px;">
        <p>Explore and analyze web traffic data with Chrona!</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <div style="width: 200px;">
        <p>Look at traffic predictions by selecting 'SARIMAX' in the sidebar 
    or check autocorrelation with 'ACF Plot'.</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.sidebar.columns(2)

# Display buttons with tooltips on the sidebar
button1 = st.sidebar.button('📊 SARIMAX')
button2 = st.sidebar.button('📈 ACF Plot')


if button1:
    # Train and predict with SARIMA
    y_pred_sarima, residuals = train_and_predict_sarima(X_train, y_train, X_test)
    # Display SARIMA visualizations here
    plt.figure()
    plt.plot(y_test, color='red', label='Real Web View')
    plt.plot(y_pred_sarima, color='blue', label='Predicted Web View (SARIMAX)')
    plt.title('Web View Forecasting - SARIMAX')
    plt.xlabel('Number of Days from Start')
    plt.ylabel('Web View')
    plt.legend()
    st.pyplot(plt)  # Display the plot in Streamlit app

if button2:
    y_pred_sarima, residuals = train_and_predict_sarima(X_train, y_train, X_test)
    # ACF plot for residuals
    plt.figure()
    plot_acf(residuals, lags=20, title='Autocorrelation Plot for Residuals')
    st.pyplot(plt)  # Display the ACF plot in Streamlit app
