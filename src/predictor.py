# src/predictor.py
import pandas as pd
import pickle
from prophet import Prophet
import logging
import os

# --- 路径配置保持不变 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'traffic_5min.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'final_model.pkl')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_advanced_model():
    """
    升级版模型工厂：
    1. 区分 '工作日' 和 '周末' 的不同日周期模式 (Conditional Seasonality)。
    2. 放宽正则化参数，允许更剧烈的波动。
    """
    model = Prophet(
        daily_seasonality=False,      # 必须关闭默认，我们要手动拆分
        weekly_seasonality=True,      # 保留大趋势的周周期
        seasonality_mode='additive',
        
        # --- 策略 B: 释放灵活性 ---
        changepoint_prior_scale=0.1,  # (原0.05) 允许趋势变得更灵活
        seasonality_prior_scale=15.0  # (原10.0) 允许季节性波动的幅度更大
    )
    
    # --- 策略 A: 添加条件季节性 ---
    # 定义工作日模式 (Workdays): 尖峰很高，且有早晚高峰，给高阶数 N=20
    model.add_seasonality(
        name='workday_daily', 
        period=1, 
        fourier_order=20,  # 提高阶数以捕捉更尖锐的峰值
        condition_name='is_workday' # 只有满足此条件时才应用
    )
    
    # 定义周末模式 (Weekends): 比较平缓，给低阶数 N=10 避免过拟合
    model.add_seasonality(
        name='weekend_daily', 
        period=1, 
        fourier_order=10, 
        condition_name='is_weekend'
    )
    
    return model

def prepare_data(df):
    """
    特征工程：为条件季节性添加布尔列
    """
    df = df.copy()
    df['ds'] = pd.to_datetime(df['Time'])
    df['y'] = df['people_in']
    
    # 构造条件列
    # dayofweek: 0=Mon, 6=Sun
    df['is_weekend'] = df['ds'].dt.dayofweek >= 5
    df['is_workday'] = ~df['is_weekend']
    
    return df

def train_and_save_final_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件未找到: {DATA_PATH}")

    logger.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # --- 关键修改：应用特征工程 ---
    df_prepared = prepare_data(df)
    
    logger.info("Initializing ADVANCED model with Conditional Seasonality...")
    model = create_advanced_model()
    
    logger.info("Fitting model on full dataset...")
    model.fit(df_prepared)
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    logger.info(f"Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info("Done.")

if __name__ == "__main__":
    train_and_save_final_model()