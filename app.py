# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:02:30 2023

@author: Punam
"""

import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

st.title('Forecasting Cement Sales')
st.write("Import the time series CSV file") 
uploaded_file = st.file_uploader(" ", type=['csv'])

if uploaded_file is not None:     
     data = pd.read_csv(uploaded_file)
     st.write(uploaded_file)
     model = ARIMA(data['Sales_MT'], order=(6,1,6))
     model_fit = model.fit()
     pred = model_fit.predict(start = len(data), end = len(data)+23)
     st.subheader(" ARIMA Model")
     st.write('Forecasted Sales Values', pred)
     st.title("Actual values vs Forecasted values")
     fig, ax = plt.subplots()
     ax.plot(data.Sales_MT,'-b', label='Actual Value')
     ax.plot(pred, '-r', label = 'Predicted value')
     ax.legend();
     st.pyplot(fig)
     
     
     st.subheader("Thanks for Visit")
  
   