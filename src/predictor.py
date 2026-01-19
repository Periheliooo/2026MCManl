# src/predictor.py
import pandas as pd
import pickle
from prophet import Prophet
import logging
import os

# --- 路径配置 (Path Configuration) ---
# 获取当前脚本(predictor.py)所在的绝对路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (src的上一级)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 定义标准路径
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'traffic_5min.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'final_model.pkl')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_standard_model():
    """
    模型工厂函数：定义标准化的 Prophet 模型配置。
    确保验证集和测试集的模型结构完全一致。
    """
    model = Prophet(
        daily_seasonality=False,     # 关闭默认日周期
        weekly_seasonality=True,     # 开启周周期
        seasonality_mode='additive', # 加法模式
        changepoint_prior_scale=0.05 # 趋势灵活性正则化
    )
    
    # 手动添加高阶日周期 (Order=15)，用于拟合尖锐的早晚高峰
    model.add_seasonality(
        name='high_res_daily', 
        period=1, 
        fourier_order=15 
    )
    
    return model

def train_and_save_final_model():
    """
    读取全量数据训练，并将模型保存到 models/ 文件夹。
    """
    # 1. 检查数据文件是否存在
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件未找到: {DATA_PATH}")

    logger.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # 格式转换
    df['ds'] = pd.to_datetime(df['Time'])
    df['y'] = df['people_in']
    
    logger.info("Initializing standard model...")
    model = create_standard_model()
    
    logger.info("Fitting model on full dataset (All 30 days)...")
    model.fit(df)
    
    # 2. 自动创建模型保存目录 (如果不存在)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        logger.info(f"Created directory: {MODEL_DIR}")
    
    logger.info(f"Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info("Done. Model is ready for production.")

if __name__ == "__main__":
    train_and_save_final_model()