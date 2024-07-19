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
    train_df = st.session_state['train_df']
    test_df = st.session_state['test_df']
    value_column = st.session_state['value_column']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if model_type in ["ARIMA", "SARIMA"]:
        results = st.session_state['model_results']
        
        # 訓練データの予測
        train_predict = results.get_prediction(start=train_df.index[0], end=train_df.index[-1])
        train_forecast = train_predict.predicted_mean
        
        # テストデータの予測
        forecast = results.get_forecast(steps=len(test_df))
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
        
        # プロット
        ax.plot(train_df.index, train_df[value_column], label='訓練データ（実測）', color='blue')
        ax.plot(train_df.index, train_forecast, label='訓練データ（予測）', color='lightblue')
        ax.plot(test_df.index, test_df[value_column], label='テストデータ（実測）', color='green')
        ax.plot(forecast_mean.index, forecast_mean.values, label='テストデータ（予測）', color='lightgreen')
        ax.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='lightgreen', alpha=0.3)
        
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
        
        # プロット
        ax.plot(train_df.index, train_df[value_column], label='訓練データ（実測）', color='blue')
        ax.plot(train_forecast['ds'], train_forecast['yhat'], label='訓練データ（予測）', color='lightblue')
        ax.plot(test_df.index, test_df[value_column], label='テストデータ（実測）', color='green')
        ax.plot(test_forecast['ds'], test_forecast['yhat'], label='テストデータ（予測）', color='lightgreen')
        ax.fill_between(test_forecast['ds'], test_forecast['yhat_lower'], test_forecast['yhat_upper'], color='lightgreen', alpha=0.3)
        
        forecast_df = pd.DataFrame({
            'actual': test_df[value_column],
            'predicted': test_forecast['yhat'],
            'lower_ci': test_forecast['yhat_lower'],
            'upper_ci': test_forecast['yhat_upper']
        })

    # グラフの設定
    ax.set_title("訓練データとテストデータの予測結果")
    ax.legend()
    ax.axvline(x=test_df.index[0], color='red', linestyle='--', label='訓練/テスト分割点')
    ax.text(test_df.index[0], ax.get_ylim()[1], '  テスト期間開始', color='red', va='top')
    
    st.pyplot(fig)
    
    st.subheader("予測結果データ")
    st.write(forecast_df)
    
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