# pages/2_time_series_analysis.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px

st.title("時系列分析")

if 'preprocessed_data' not in st.session_state:
    st.warning("まずデータの前処理を行ってください。")
else:
    df = st.session_state['preprocessed_data'].copy()
    
    # 時系列データの選択
    col1, col2 = st.columns(2)
    with col1:
        time_column = st.selectbox("時間列を選択 (任意)", df.columns, key="time_col")
    with col2:
        # time_column を除外したカラムのリストを作成
        value_column_options = [col for col in df.columns if col != time_column]
        value_column = st.selectbox("分析する列を選択", value_column_options, key="value_col")

    # 元のindexを保持するかどうか
    keep_original_index = st.checkbox("元のindexを保持する", value=False)

    if time_column:  # time_column が選択されている場合のみ処理
        try:
            df[time_column] = pd.to_datetime(df[time_column])
            if not keep_original_index:
                df = df.set_index(time_column)
        except:
            st.warning("選択されたカラムは時系列データに変換できませんでした。元のデータフレームをそのまま使用します。")

    series = df[value_column]

    st.subheader("時系列プロット")
    fig = px.line(df, x=df.index, y=value_column, title=f"{value_column} の時系列プロット")
    # fig.update_layout(xaxis_title="時間", yaxis_title=value_column)
    fig.update_layout(
        xaxis_title="時間",
        yaxis_title=value_column,
        xaxis_tickformat="%Y-%m-%d"  # YYYY-MM-DD 形式に変更
        )
    st.plotly_chart(fig)
    
    st.subheader("定常性の確認 (ADF検定)")
    result = adfuller(series.dropna())

    st.write(f'ADF統計量: {result[0]:.4f}')  # 小数点以下4桁まで表示
    st.write(f'p値: {result[1]:.4f}')  # 小数点以下4桁まで表示

    # p値が0.05より小さい場合は定常性があると判定
    if result[1] < 0.05:
        st.write("判定結果: 定常性あり")
    else:
        st.write("判定結果: 定常性なし")
    
    st.subheader("自己相関・偏自己相関")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    plot_acf(series.dropna(), ax=ax1)
    plot_pacf(series.dropna(), ax=ax2)
    st.pyplot(fig)
    
    st.subheader("季節性分解")
    period = st.slider("季節性の周期", 2, 365, 12)
    decomposition = seasonal_decompose(series.dropna(), model='additive', period=period)
    fig = decomposition.plot()
    st.pyplot(fig)

    st.session_state['analyzed_data'] = df
    st.success("時系列分析が完了しました。")