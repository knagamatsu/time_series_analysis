# pages/4_prediction.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

st.title("予測")

if 'model' not in st.session_state:
    st.warning("まずモデリングを行ってください。")
else:
    model = st.session_state['model']
    model_type = st.session_state['model_type']
    train_df = st.session_state['train_df']  # 訓練データを session_state から取得
    test_df = st.session_state['test_df']
    value_column = st.session_state['value_column']
    
    if model_type in ["ARIMA", "SARIMA"]:
        results = st.session_state['model_results']
        
        # 訓練データの予測
        train_predict = results.get_prediction(start=train_df.index[0], end=train_df.index[-1])
        train_forecast = train_predict.predicted_mean
        
        # テストデータの予測
        forecast = results.get_forecast(steps=len(test_df))
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
        
        # 予測結果のプロット
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # 訓練データのプロット
        ax1.plot(train_df.index, train_df[value_column], label='訓練データ（実測）')
        ax1.plot(train_df.index, train_forecast, label='訓練データ（予測）')
        ax1.set_title("訓練データの予測結果")
        ax1.legend()
        
        # テストデータのプロット
        ax2.plot(test_df.index, test_df[value_column], label='テストデータ（実測）')
        ax2.plot(forecast_mean.index, forecast_mean.values, label='テストデータ（予測）')
        ax2.fill_between(forecast_ci.index, 
                         forecast_ci.iloc[:, 0], 
                         forecast_ci.iloc[:, 1], 
                         alpha=0.3)
        ax2.set_title("テストデータの予測結果")
        ax2.legend()
        
        st.pyplot(fig)
        
        forecast_df = pd.DataFrame({
            'actual': test_df[value_column],
            'predicted_mean': forecast_mean,
            'lower_ci': forecast_ci.iloc[:, 0],
            'upper_ci': forecast_ci.iloc[:, 1]
        })
        
    elif model_type == "Prophet":
        # 訓練データの予測
        train_future = pd.DataFrame({'ds': train_df.index})
        for regressor in st.session_state['regressors']:
            train_future[regressor] = train_df[regressor].values
        train_forecast = model.predict(train_future)
        
        # テストデータの予測
        test_future = pd.DataFrame({'ds': test_df.index})
        for regressor in st.session_state['regressors']:
            test_future[regressor] = test_df[regressor].values
        test_forecast = model.predict(test_future)
        # 予測結果のプロット
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # 訓練データのプロット
        ax1.plot(train_df.index, train_df[value_column], label='訓練データ（実測）')
        ax1.plot(train_forecast['ds'], train_forecast['yhat'], label='訓練データ（予測）')
        ax1.fill_between(train_forecast['ds'], 
                         train_forecast['yhat_lower'], 
                         train_forecast['yhat_upper'], 
                         alpha=0.3)
        ax1.set_title("訓練データの予測結果")
        ax1.legend()
        
        # テストデータのプロット
        ax2.plot(test_df.index, test_df[value_column], label='テストデータ（実測）')
        ax2.plot(test_forecast['ds'], test_forecast['yhat'], label='テストデータ（予測）')
        ax2.fill_between(test_forecast['ds'], 
                         test_forecast['yhat_lower'], 
                         test_forecast['yhat_upper'], 
                         alpha=0.3)
        ax2.set_title("テストデータの予測結果")
        ax2.legend()
        
        st.pyplot(fig)
        
        forecast_df = pd.DataFrame({
            'actual': test_df[value_column].values,
            'predicted': test_forecast['yhat'].values,
            'lower_ci': test_forecast['yhat_lower'].values,
            'upper_ci': test_forecast['yhat_upper'].values
        })
    
    st.subheader("予測結果データ")
    st.write(forecast_df)
    forecast_df.dropna(inplace=True)
    
    # 予測精度の評価
    mae = mean_absolute_error(forecast_df['actual'], forecast_df['predicted_mean'] if model_type in ["ARIMA", "SARIMA"] else forecast_df['predicted'])
    rmse = np.sqrt(mean_squared_error(forecast_df['actual'], forecast_df['predicted_mean'] if model_type in ["ARIMA", "SARIMA"] else forecast_df['predicted']))
    
    st.subheader("予測精度")
    st.write(f"平均絶対誤差 (MAE): {mae:.4f}")
    st.write(f"二乗平均平方根誤差 (RMSE): {rmse:.4f}")
    
    # 予測結果のダウンロード
    csv = forecast_df.to_csv(index=True)
    st.download_button(
        label="予測結果をCSVでダウンロード",
        data=csv,
        file_name="forecast.csv",
        mime="text/csv",
    )

    st.success("予測が完了しました。")