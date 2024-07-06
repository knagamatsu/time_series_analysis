# pages/time_series_analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

def time_series_analysis():
    st.title("時系列分析")
    
    if 'preprocessed_data' not in st.session_state:
        st.warning("まずデータの前処理を行ってください。")
        return
    
    df = st.session_state['preprocessed_data'].copy()
    
    # 時系列データの選択
    time_column = st.selectbox("時間列を選択", df.columns)
    value_column = st.selectbox("分析する列を選択", df.columns)
    
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.set_index(time_column)
    series = df[value_column]
    
    st.subheader("時系列プロット")
    fig, ax = plt.subplots()
    plt.rcParams['font.family'] = 'IPAexGothic'
    ax.plot(series)
    ax.set_title(f"{value_column}の時系列プロット")
    st.pyplot(fig)
    
    st.subheader("定常性の確認 (ADF検定)")
    result = adfuller(series.dropna())
    st.write(f'ADF統計量: {result[0]}')
    st.write(f'p値: {result[1]}')
    st.write("臨界値:")
    for key, value in result[4].items():
        st.write(f'   {key}: {value}')
    
    st.subheader("自己相関・偏自己相関")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    plot_acf(series.dropna(), ax=ax1)
    plot_pacf(series.dropna(), ax=ax2)
    st.pyplot(fig)
    
    st.subheader("季節性分解")
    decomposition = seasonal_decompose(series.dropna(), model='additive', period=st.slider("季節性の周期", 2, 365, 12))
    fig = decomposition.plot()
    st.pyplot(fig)

if __name__ == "__main__":
    time_series_analysis()