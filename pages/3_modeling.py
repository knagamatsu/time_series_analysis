# pages/3_modeling.py
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

st.title("モデリング")

if 'analyzed_data' not in st.session_state:
    st.warning("まず時系列分析を行ってください。")
else:
    df = st.session_state['analyzed_data'].copy()

    # value_column の選択
    value_column = st.selectbox("予測する列を選択", df.select_dtypes(include=[np.number]).columns)
    series = df[value_column]

    # モデル選択
    model_type = st.selectbox("モデルを選択", ["Prophet", "ARIMA", "SARIMA"])

    # Prophetの場合はindexをリセット
    if model_type == "Prophet":
        if not isinstance(df.index, pd.DatetimeIndex):
            st.error('Prophetモデルでは時系列の列を選択してください。')
            st.stop()
        else:
            # Prophet用のデータフレームを作成
            df_prophet = pd.DataFrame({
                'ds': df.index,  # indexを"ds"カラムとして使用
                'y': series  # 予測対象列を"y"カラムとして使用
            })
        model = Prophet()
        model.fit(df_prophet)  # df_prophetを使って学習

    elif model_type == "ARIMA":
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

    st.session_state['model'] = model
    st.session_state['model_type'] = model_type
    st.session_state['model_results'] = results if model_type in ["ARIMA", "SARIMA"] else None

    if model_type in ["ARIMA", "SARIMA"]:
        st.subheader("モデルサマリー")
        st.text(results.summary())

    st.success("モデルの学習が完了しました。")