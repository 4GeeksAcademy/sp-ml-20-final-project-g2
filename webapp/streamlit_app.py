from pickle import load
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title='Pharmacy Demand Forecast', layout='wide')

st.title('📦 Pharmacy Demand Forecast')

model = load(open('models/pharmacy-demand.pkl', 'rb'))

df = pd.read_csv('../data/raw/farmacia-datos.csv', sep=';',encoding='latin-1')
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Features
features = [
    'week_day', 'month', 'month_day',
    'weekend', 'beginning_month',
    'cant_lag_1', 'cant_lag_7',
    'cant_lag_14', 'cant_lag_30',
    'rolling_7', 'rolling_30'
]

# Sidebar
st.sidebar.header('Filters')
# Aqui debo leer el archivo json
product = st.sidebar.selectbox('Select product', df['groups'].unique())

# Filter product
df_prod = df[df['groups'] == product].sort_values('Fecha')

# Predict
df_prod['prediction'] = model.predict(df_prod[features])

# Plot
st.subheader(f'Demand for {product}')
st.line_chart(
    df_prod.set_index('Fecha')[['Cant', 'prediction']]
)

# Metrics
mae = (df_prod['Cant'] - df_prod['prediction']).abs().mean()
st.metric("MAE", round(mae, 2))