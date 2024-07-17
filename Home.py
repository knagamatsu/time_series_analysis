import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os
import pandas as pd

st.set_page_config(page_title="化学工場時系列データ解析", layout="wide")

def set_plot_style():
    system = platform.system()
    font_paths = []
    
    if system == 'Windows':
        font_paths = [
            'C:\\Windows\\Fonts\\meiryo.ttc',
            'C:\\Windows\\Fonts\\yugothic.ttc',
            'C:\\Windows\\Fonts\\msgothic.ttc'
        ]
    elif system == 'Darwin':  # macOS
        font_paths = [
            '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',
            '/System/Library/Fonts/AppleGothic.ttf',
            '/Library/Fonts/Osaka.ttf'
        ]
    else:  # Linux
        font_paths = [
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/vlgothic/VL-Gothic-Regular.ttf'
        ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=font_path).get_name()]
            return
    
    st.warning("適切な日本語フォントが見つかりませんでした。テキストが正しく表示されない可能性があります。")

set_plot_style()

st.title("時系列データ解析アプリ")

st.write("""
このアプリケーションは、化学工場の時系列データを解析するためのツールです。
サイドバーのナビゲーションを使用して、各機能にアクセスしてください。

1. データ読み込み: このページでCSVまたはExcelファイルからデータをアップロードします。
2. データ前処理: 欠損値処理、外れ値検出、スケーリングを行います。
3. 時系列分析: 定常性チェック、自己相関分析、季節性分解を行います。
4. モデリング: ARIMA、SARIMA、Prophetモデルを適用します。
5. 予測: 将来の値を予測し、可視化します。

各ページの指示に従って、段階的にデータ解析を進めてください。
""")

# データ読み込み機能をここに実装
def load_data():
    uploaded_file = st.file_uploader("CSVまたはExcelファイルをアップロードしてください", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                encoding_list = ['utf-8', 'shift-jis', 'cp932']
                for encoding in encoding_list:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                    except Exception as e:
                        print(e)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state['data'] = df
            st.success("データが正常に読み込まれました。")
            
            st.subheader("データプレビュー")
            st.write(df.head())
            
            st.subheader("基本統計情報")
            st.write(df.describe())
            
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
    else:
        st.info("ファイルをアップロードしてください。")

load_data()