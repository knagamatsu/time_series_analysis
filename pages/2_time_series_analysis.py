import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.decomposition import PCA

def plot_time_series(df, value_column):
    fig = px.line(df, x=df.index, y=value_column, title=f"{value_column} の時系列プロット")
    fig.update_layout(
        xaxis_title="時間",
        yaxis_title=value_column,
        xaxis_tickformat="%Y-%m-%d"
    )
    return fig

def perform_adf_test(series):
    result = adfuller(series.dropna())
    return result[0], result[1]

def plot_acf_pacf(series):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    plot_acf(series.dropna(), ax=ax1)
    plot_pacf(series.dropna(), ax=ax2)
    ax1.set_title("自己相関")
    ax2.set_title("偏自己相関")
    return fig

def plot_seasonal_decompose(series, period):
    decomposition = seasonal_decompose(series.dropna(), model='additive', period=period)
    fig = decomposition.plot()
    return fig

def perform_pca(df, pca_columns, value_column):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[pca_columns])
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df.index = df.index
    pca_df[value_column] = df[value_column]
    return pca, pca_df

def plot_pca_scatter(pca_df, value_column):
    fig = px.scatter(pca_df, x='PC1', y='PC2', color=value_column,
                     title='PCA Visualization',
                     labels={value_column: f'{value_column} (color)'},
                     color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout(
        xaxis_title="First Principal Component",
        yaxis_title="Second Principal Component"
    )
    return fig

def plot_cumulative_variance_ratio(pca):
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    fig = px.bar(x=range(1, len(cumulative_variance_ratio)+1), y=cumulative_variance_ratio,
                  labels={'x': '主成分数', 'y': '累積寄与率'},
                  title='累積寄与率')
    fig.add_hline(y=0.8, line_dash="dash", line_color="red", annotation_text="80% threshold")
    return fig

def plot_pca_variance(pca, n_components=10):
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    n_components = min(n_components, len(explained_variance_ratio))
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 寄与率（棒グラフ）
    fig.add_trace(
        go.Bar(x=list(range(1,  n_components + 1)), 
               y=explained_variance_ratio, 
               name="寄与率"),
        secondary_y=False,
    )
    
    # 累積寄与率（線グラフ）
    fig.add_trace(
        go.Scatter(x=list(range(1, n_components + 1)), 
                   y=cumulative_variance_ratio, 
                   name="累積寄与率"),
        secondary_y=True,
    )
    
    fig.update_layout(
        title_text="主成分の寄与率と累積寄与率",
        xaxis_title="主成分",
    )
    
    fig.update_yaxes(title_text="寄与率", secondary_y=False, range=[0, 1])
    fig.update_yaxes(title_text="累積寄与率", secondary_y=True, range=[0, 1])
    
    # 80%のラインを追加
    fig.add_hline(y=0.8, line_dash="dash", line_color="red", secondary_y=True)
    
    return fig

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
        value_column_options = [col for col in df.columns if col != time_column]
        value_column = st.selectbox("分析する列を選択", value_column_options, key="value_col")

    keep_original_index = st.checkbox("元のindexを保持する", value=False)
    # 欠損を削除してインデックスを振りなおす。
    drop_na = st.checkbox("欠損値を削除してインデックスを振りなおす", value=False)
    if drop_na:
        df = df.dropna().reset_index(drop=True)

    if time_column:
        try:
            df[time_column] = pd.to_datetime(df[time_column])
            if not keep_original_index:
                df = df.set_index(time_column)
                
        except:
            st.warning("選択されたカラムは時系列データに変換できませんでした。元のデータフレームをそのまま使用します。")

    series = df[value_column]

    st.subheader("時系列プロットとPCA可視化")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_time_series(df, value_column), use_container_width=True)

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    pca_columns = [col for col in numeric_columns if col != value_column]
    
    if len(pca_columns) > 1:
        pca, pca_df = perform_pca(df, pca_columns, value_column)
        with col2:
            st.plotly_chart(plot_pca_scatter(pca_df, value_column), use_container_width=True)
    else:
        with col2:
            st.warning("PCAを実行するには少なくとも2つの数値列が必要です（解析対象列を除く）。")
    
    st.subheader("定常性の確認 (ADF検定)")
    adf_statistic, p_value = perform_adf_test(series)
    st.write(f'ADF統計量: {adf_statistic:.4f}')
    st.write(f'p値: {p_value:.4f}')
    st.write("判定結果: " + ("定常性あり" if p_value < 0.05 else "定常性なし"))
    
    st.subheader("自己相関・偏自己相関")
    st.pyplot(plot_acf_pacf(series))
    
    st.subheader("季節性分解")
    period = st.slider("季節性の周期", 2, 365, 12)
    st.pyplot(plot_seasonal_decompose(series, period))
    
    col1, col2 = st.columns(2)
    with col1:
        if len(pca_columns) > 1:
            st.subheader("主成分分析 (PCA)")
            explained_variance_ratio = pca.explained_variance_ratio_
            st.write(f"PC1の寄与率: {explained_variance_ratio[0]:.2f}")
            st.write(f"PC2の寄与率: {explained_variance_ratio[1]:.2f}")

            st.subheader("主成分の構成")
            component_df = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=pca_columns)
            st.dataframe(component_df)

            st.subheader("累積寄与率")
            st.plotly_chart(plot_pca_variance(pca), use_container_width=True)

    st.session_state['analyzed_data'] = df
    st.success("時系列分析が完了しました。")
    
    st.subheader("分析結果")
    st.dataframe(df)
    
    csv = df.to_csv()
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='dataframe.csv',
        mime='text/csv',
    )