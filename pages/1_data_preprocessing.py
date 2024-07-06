# pages/1_data_preprocessing.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

st.title("データ前処理")

if 'data' not in st.session_state:
    st.warning("まずホームページでデータを読み込んでください。")
else:
    df = st.session_state['data'].copy()

    st.subheader("欠損値の処理")
    missing_method = st.selectbox("欠損値の処理方法", ["削除", "平均値で補完", "中央値で補完", "線形補間"])
    
    if missing_method == "削除":
        df = df.dropna()
    elif missing_method == "平均値で補完":
        df = df.fillna(df.mean())
    elif missing_method == "中央値で補完":
        df = df.fillna(df.median())
    elif missing_method == "線形補間":
        df = df.interpolate()
    
    st.subheader("外れ値の検出")
    z_threshold = st.slider("Z-scoreのしきい値", 2.0, 4.0, 3.0, 0.1)
    
    for column in df.select_dtypes(include=[np.number]).columns:
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        df[f"{column}_is_outlier"] = z_scores > z_threshold
    
    st.subheader("スケーリング")
    scaling_method = st.selectbox("スケーリング方法", ["なし", "Min-Max", "標準化"])
    
    if scaling_method == "Min-Max":
        scaler = MinMaxScaler()
        df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    elif scaling_method == "標準化":
        scaler = StandardScaler()
        df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    
    st.session_state['preprocessed_data'] = df
    
    st.subheader("前処理後のデータプレビュー")
    st.write(df.head())
    
    st.success("データの前処理が完了しました。")