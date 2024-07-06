# pages/prediction.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

def prediction():
    st.title("予測")
    
    if 'model' not in st.session_state:
        st.warning("まずモデリングを行ってください。")
        return
    
    model = st.session_state['model']
    model_type = st.session_state['model_type']
    
    # 予測期間の設定
    forecast_period = st.slider("予測期間（日数）", 1, 365, 30)
    
    if model_type in ["ARIMA", "SARIMA"]:
        forecast = model.forecast(steps=forecast_period)
        
        # 予測結果のプロット
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(forecast.index, forecast.values, label='予測')
        ax.fill_between(forecast.index, 
                        forecast.conf_int().iloc[:, 0], 
                        forecast.conf_int().iloc[:, 1], 
                        alpha=0.3)
        ax.set_title("予測結果")
        ax.legend()
        st.pyplot(fig)
        
    elif model_type == "Prophet":
        future = model.make_future_dataframe(periods=forecast_period)
        forecast = model.predict(future)
        
        # 予測結果のプロット
        fig = model.plot(forecast)
        ax = fig.gca()
        ax.set_title("予測結果")
        st.pyplot(fig)
        
        # コンポーネントのプロット
        fig = model.plot_components(forecast)
        st.pyplot(fig)
    
    st.subheader("予測結果データ")
    st.write(forecast.tail())
    
    # 予測結果のダウンロード
    csv = forecast.to_csv(index=True)
    st.download_button(
        label="予測結果をCSVでダウンロード",
        data=csv,
        file_name="forecast.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    prediction()