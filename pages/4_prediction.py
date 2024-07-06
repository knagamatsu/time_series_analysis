# pages/4_prediction.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

st.title("予測")

if 'model' not in st.session_state:
    st.warning("まずモデリングを行ってください。")
else:
    model = st.session_state['model']
    model_type = st.session_state['model_type']
    
    # 予測期間の設定
    forecast_period = st.slider("予測期間（日数）", 1, 365, 30)
    
    if model_type in ["ARIMA", "SARIMA"]:
        results = st.session_state['model_results']
        forecast = results.get_forecast(steps=forecast_period)
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
        
        # 予測結果のプロット
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(forecast_mean.index, forecast_mean.values, label='予測')
        ax.fill_between(forecast_ci.index, 
                        forecast_ci.iloc[:, 0], 
                        forecast_ci.iloc[:, 1], 
                        alpha=0.3)
        ax.set_title("予測結果")
        ax.legend()
        st.pyplot(fig)
        
        forecast_df = pd.DataFrame({
            'predicted_mean': forecast_mean,
            'lower_ci': forecast_ci.iloc[:, 0],
            'upper_ci': forecast_ci.iloc[:, 1]
        })
        
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
        
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_period)
    
    st.subheader("予測結果データ")
    st.write(forecast_df)
    
    # 予測結果のダウンロード
    csv = forecast_df.to_csv(index=True)
    st.download_button(
        label="予測結果をCSVでダウンロード",
        data=csv,
        file_name="forecast.csv",
        mime="text/csv",
    )

    st.success("予測が完了しました。")