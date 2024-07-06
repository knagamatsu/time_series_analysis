import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os

# Streamlit設定を最初に行う
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

st.sidebar.title("ナビゲーション")
page = st.sidebar.radio("ページ選択", ["データ読み込み", "データ前処理", "時系列分析", "モデリング", "予測"])

if page == "データ読み込み":
    import pages._1_data_loading as data_loading
    pages.data_loading.load_data()

elif page == "データ前処理":
    import pages._2_data_preprocessing as data_preprocessing
    pages.data_preprocessing.preprocess_data()

elif page == "時系列分析":
    import pages._3_time_series_analysis as time_series_analysis
    pages.time_series_analysis.time_series_analysis()

elif page == "モデリング":
    import pages._4_modeling as modeling
    pages.modeling.modeling()

elif page == "予測":
    import pages._5_prediction as prediction
    pages.prediction.prediction()