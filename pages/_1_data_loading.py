# pages/data_loading.py
import streamlit as st
import pandas as pd
import io

def load_data():
    st.title("データ読み込み")
    
    uploaded_file = st.file_uploader("CSVまたはExcelファイルをアップロードしてください", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                encoding_list = ['utf-8', 'shift-jis', 'cp932', 'euc-jp']
                for encoding in encoding_list:
                    try:
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        break
                    except Exception:
                        continue
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state['data'] = df
            st.success("データが正常に読み込まれました。")
            
            st.subheader("データプレビュー")
            st.write(df.head())
            
            st.subheader("基本統計情報")
            st.write(df.describe())
            
            st.subheader("データ情報")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        
        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
    
    else:
        st.info("ファイルをアップロードしてください。")

if __name__ == "__main__":
    load_data()