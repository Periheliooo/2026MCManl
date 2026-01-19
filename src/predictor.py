# src/predictor.py
import pandas as pd
import pickle
from prophet import Prophet
import logging
import os

# --- 路径配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'traffic_5min.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

# 定义两个模型的保存路径
MODEL_PATH_WORK = os.path.join(MODEL_DIR, 'model_workday.pkl')
MODEL_PATH_REST = os.path.join(MODEL_DIR, 'model_weekend.pkl')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_single_mode_model(is_workday_mode=True):
    """
    创建单模式模型。
    不需要复杂的条件语句了，因为每个模型只吃属于它自己的数据。
    """
    model = Prophet(
        daily_seasonality=False,   # 关闭默认，手动添加
        weekly_seasonality=False,  # 不需要周周期，因为我们已经按天拆分了
        seasonality_mode='additive',
        changepoint_prior_scale=0.05 
    )
    
    # 设定傅里叶阶数
    # 工作日流量复杂，阶数高一点 (12)
    # 周末流量简单，阶数低一点 (6)
    order = 12 if is_workday_mode else 6
    
    model.add_seasonality(
        name='daily', 
        period=1, 
        fourier_order=order
    )
    
    return model

def prepare_data(df):
    df = df.copy()
    df['ds'] = pd.to_datetime(df['Time'])
    df['y'] = df['people_in']
    # 标记周末 (5=Sat, 6=Sun)
    df['is_weekend'] = df['ds'].dt.dayofweek >= 5
    return df

def train_and_save_final_model():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件未找到: {DATA_PATH}")

    logger.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df = prepare_data(df)
    
    # --- 核心修改：拆分数据集 ---
    df_work = df[~df['is_weekend']].copy() # 工作日数据
    df_rest = df[df['is_weekend']].copy()  # 周末数据
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 1. 训练工作日模型
    logger.info(f"Training Workday Model (Samples: {len(df_work)})...")
    m_work = create_single_mode_model(is_workday_mode=True)
    m_work.fit(df_work)
    with open(MODEL_PATH_WORK, 'wb') as f:
        pickle.dump(m_work, f)
        
    # 2. 训练周末模型
    logger.info(f"Training Weekend Model (Samples: {len(df_rest)})...")
    m_rest = create_single_mode_model(is_workday_mode=False)
    m_rest.fit(df_rest)
    with open(MODEL_PATH_REST, 'wb') as f:
        pickle.dump(m_rest, f)
    
    logger.info("Done. Two separate models saved.")

if __name__ == "__main__":
    train_and_save_final_model()