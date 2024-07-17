# pages/3_modeling.py
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.model_selection import train_test_split

st.title("モデリング")

if 'analyzed_data' not in st.session_state:
    st.warning("まず時系列分析を行ってください。")
else:
    df = st.session_state['analyzed_data'].copy()

    # value_column の選択
    value_column = st.selectbox("予測する列を選択", df.select_dtypes(include=[np.number]).columns)
    
    # モデル選択
    model_type = st.selectbox("モデルを選択", ["Prophet", "ARIMA", "SARIMA"])

    # テストデータの割合を選択
    test_size = st.slider("テストデータの割合", 0.1, 0.5, 0.2, 0.05)

    # データを訓練用とテスト用に分割
    train_df, test_df = train_test_split(df, test_size=test_size, shuffle=False)

    if model_type == "Prophet":
        if not isinstance(df.index, pd.DatetimeIndex):
            st.error('Prophetモデルでは時系列の列を選択してください。')
            st.stop()
        else:
            # Prophet用のデータフレームを作成
            train_prophet = pd.DataFrame({
                'ds': train_df.index,
                'y': train_df[value_column]
            })
            
            # 説明変数の選択
            regressors = st.multiselect("説明変数を選択", df.select_dtypes(include=[np.number]).columns.drop(value_column))

            for regressor in regressors:
                train_prophet[regressor] = train_df[regressor]

            model = Prophet()
            for regressor in regressors:
                model.add_regressor(regressor)
            
            model.fit(train_prophet)
            st.session_state['train_df'] = train_df

    elif model_type == "ARIMA":
        p = st.slider("ARパラメータ (p)", 0, 5, 1)
        d = st.slider("差分パラメータ (d)", 0, 2, 1)
        q = st.slider("MAパラメータ (q)", 0, 5, 1)

        model = ARIMA(train_df[value_column], order=(p, d, q))
        results = model.fit()
        st.session_state['train_df'] = train_df

    elif model_type == "SARIMA":
        p = st.slider("ARパラメータ (p)", 0, 5, 1)
        d = st.slider("差分パラメータ (d)", 0, 2, 1)
        q = st.slider("MAパラメータ (q)", 0, 5, 1)
        P = st.slider("季節ARパラメータ (P)", 0, 2, 1)
        D = st.slider("季節差分パラメータ (D)", 0, 1, 1)
        Q = st.slider("季節MAパラメータ (Q)", 0, 2, 1)
        m = st.slider("季節周期 (m)", 2, 12, 12)

        model = SARIMAX(train_df[value_column], order=(p, d, q), seasonal_order=(P, D, Q, m))
        results = model.fit()
        st.session_state['train_df'] = train_df

    st.session_state['model'] = model
    st.session_state['model_type'] = model_type
    st.session_state['model_results'] = results if model_type in ["ARIMA", "SARIMA"] else None
    
    st.session_state['test_df'] = test_df
    st.session_state['value_column'] = value_column
    st.session_state['regressors'] = regressors if model_type == "Prophet" else []

    if model_type in ["ARIMA", "SARIMA"]:
        st.subheader("モデルサマリー")
        st.text(results.summary())

    st.success("モデルの学習が完了しました。")