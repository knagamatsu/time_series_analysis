import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample

def generate_batch_distillation_data(n_samples=1000):
    """バッチ蒸留のサンプルデータを生成"""
    np.random.seed(42)
    
    # 変数の生成
    temperature = np.random.normal(80, 5, n_samples)
    pressure = np.random.normal(1, 0.1, n_samples)
    feed_rate = np.random.normal(100, 10, n_samples)
    catalyst_concentration = np.random.normal(0.5, 0.05, n_samples)
    
    # 不純物量の計算 (iid)
    impurity = (0.5 * temperature + 0.3 * pressure - 0.2 * feed_rate + 0.1 * catalyst_concentration + 
                np.random.normal(0, 1, n_samples))
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='H'),
        'temperature': temperature,
        'pressure': pressure,
        'feed_rate': feed_rate,
        'catalyst_concentration': catalyst_concentration,
        'impurity': impurity
    })
    
    return df

def generate_continuous_distillation_data(n_samples=1000):
    """連続蒸留のサンプルデータを生成"""
    np.random.seed(42)
    
    # 変数の生成 (自己相関あり)
    ar_params = np.array([0.8, -0.2])
    ma_params = np.array([0.5, -0.1])
    ar = np.r_[1, -ar_params]
    ma = np.r_[1, ma_params]
    
    temperature = arma_generate_sample(ar, ma, n_samples) + 80
    pressure = arma_generate_sample(ar, ma, n_samples) + 1
    feed_rate = arma_generate_sample(ar, ma, n_samples) + 100
    catalyst_concentration = arma_generate_sample(ar, ma, n_samples) + 0.5
    
    # 不純物量の計算 (自己相関あり)
    impurity = (0.5 * temperature + 0.3 * pressure - 0.2 * feed_rate + 0.1 * catalyst_concentration + 
                arma_generate_sample(ar, ma, n_samples))
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='H'),
        'temperature': temperature,
        'pressure': pressure,
        'feed_rate': feed_rate,
        'catalyst_concentration': catalyst_concentration,
        'impurity': impurity
    })
    
    return df

def generate_stock_price_data(n_samples=1000):
    """株価推移のサンプルデータを生成"""
    np.random.seed(42)
    
    # 変数の生成 (ランダムウォーク)
    market_index = np.cumsum(np.random.normal(0, 1, n_samples))
    interest_rate = np.cumsum(np.random.normal(0, 0.01, n_samples)) + 2
    exchange_rate = np.cumsum(np.random.normal(0, 0.1, n_samples)) + 100
    oil_price = np.cumsum(np.random.normal(0, 0.5, n_samples)) + 50
    
    # 株価の計算 (ランダムウォーク)
    stock_price = (0.5 * market_index + 0.2 * interest_rate - 0.1 * exchange_rate + 0.1 * oil_price + 
                   np.cumsum(np.random.normal(0, 1, n_samples))) + 100
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='D'),
        'market_index': market_index,
        'interest_rate': interest_rate,
        'exchange_rate': exchange_rate,
        'oil_price': oil_price,
        'stock_price': stock_price
    })
    
    return df

# データの生成と保存
batch_distillation_df = generate_batch_distillation_data()
batch_distillation_df.to_csv('batch_distillation_data.csv', index=False)

continuous_distillation_df = generate_continuous_distillation_data()
continuous_distillation_df.to_csv('continuous_distillation_data.csv', index=False)

stock_price_df = generate_stock_price_data()
stock_price_df.to_csv('stock_price_data.csv', index=False)

print("サンプルデータが生成され、CSVファイルとして保存されました。")