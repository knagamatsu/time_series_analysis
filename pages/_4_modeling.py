# pages/modeling.py
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

def modeling():
    st.title("モデリング")
    
    if 'preprocessed_data' not in st.session_state:
        st.warning("まずデータの前処理を行ってください。")
        return
    
    df = st.session_state['preprocessed_data'].copy()
    
    # 時系列データの選択
    time_column = st.selectbox("時間列を選択", df.columns)
    value_column = st.selectbox("予測する列を選択", df.columns)
    
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.set_index(time_column)
    series = df[value_column]
    
    # モデル選択
    model_type = st.selectbox("モデルを選択", ["ARIMA", "SARIMA", "Prophet"])
    
    if model_type == "ARIMA":
        p = st.slider("ARパラメータ (p)", 0, 5, 1)
        d = st.slider("差分パラメータ (d)", 0, 2, 1)
        q = st.slider("MAパラメータ (q)", 0, 5, 1)
        
        model = ARIMA(series, order=(p, d, q))
        results = model.fit()
        
    elif model_type == "SARIMA":
        p = st.slider("ARパラメータ (p)", 0, 5, 1)
        d = st.slider("差分パラメータ (d)", 0, 2, 1)
        q = st.slider("MAパラメータ (q)", 0, 5, 1)
        P = st.slider("季節ARパラメータ (P)", 0, 2, 1)
        D = st.slider("季節差分パラメータ (D)", 0, 1, 1)
        Q = st.slider("季節MAパラメータ (Q)", 0, 2, 1)
        m = st.slider("季節周期 (m)", 2, 12, 12)
        
        model = SARIMAX(series, order=(p, d, q), seasonal_order=(P, D, Q, m))
        results = model.fit()
        
    elif model_type == "Prophet":
        df_prophet = pd.DataFrame({'ds': series.index, 'y': series.values})
        model = Prophet()
        model.fit(df_prophet)
        
    st.session_state['model'] = model
    st.session_state['model_type'] = model_type
    st.session_state['model_results'] = results if model_type in ["ARIMA", "SARIMA"] else None
    
    if model_type in ["ARIMA", "SARIMA"]:
        st.subheader("モデルサマリー")
        st.write(results.summary())
    
    st.success("モデルの学習が完了しました。")

if __name__ == "__main__":
    modeling()