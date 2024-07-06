# app.py
import streamlit as st

st.set_page_config(page_title="化学工場時系列データ解析", layout="wide")

st.sidebar.title("ナビゲーション")
page = st.sidebar.radio("ページ選択", ["データ読み込み", "データ前処理", "時系列分析", "モデリング", "予測", "診断", "レポート生成"])

if page == "データ読み込み":
    st.title("データ読み込み")
    # データ読み込みの実装

elif page == "データ前処理":
    st.title("データ前処理")
    # データ前処理の実装

elif page == "時系列分析":
    st.title("時系列分析")
    # 時系列分析の実装

elif page == "モデリング":
    st.title("モデリング")
    # モデリングの実装

elif page == "予測":
    st.title("予測")
    # 予測の実装

elif page == "診断":
    st.title("診断")
    # 診断の実装

elif page == "レポート生成":
    st.title("レポート生成")
    # レポート生成の実装