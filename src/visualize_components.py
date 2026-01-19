# src/visualize_components.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from predictor import prepare_data

# --- 路径配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'traffic_5min.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
IMAGE_DIR = os.path.join(PROJECT_ROOT, 'images')
MODEL_PATH_WORK = os.path.join(MODEL_DIR, 'model_workday.pkl')
MODEL_PATH_REST = os.path.join(MODEL_DIR, 'model_weekend.pkl')

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def visualize_components():
    print("Loading data and models...")
    df_raw = pd.read_csv(DATA_PATH)
    df = prepare_data(df_raw)
    
    with open(MODEL_PATH_WORK, 'rb') as f:
        m_work = pickle.load(f)
    with open(MODEL_PATH_REST, 'rb') as f:
        m_rest = pickle.load(f)

    # ==========================================
    # 1. Trend Component (趋势项分解)
    # ==========================================
    print("Generating Trend Component Plot...")
    
    # 预测全量数据的 Trend
    # 技巧：分别用两个模型预测所有日期，然后根据 is_weekend 拼接趋势
    forecast_work = m_work.predict(df)
    forecast_rest = m_rest.predict(df)
    
    # 提取趋势列
    trend_combined = np.where(
        df['is_weekend'], 
        forecast_rest['trend'], 
        forecast_work['trend']
    )
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['ds'], trend_combined, color='#333333', lw=2)
    plt.title('Component 1: Trend (Underlying Baseline)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Trend Value')
    plt.grid(True, alpha=0.3)
    
    save_path_trend = os.path.join(IMAGE_DIR, 'component_1_trend.png')
    plt.savefig(save_path_trend, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path_trend}")

    # ==========================================
    # 2. Daily Component (日周期对比)
    # ==========================================
    print("Generating Daily Component Plot...")
    
    # 创建一个标准的 24小时 时间轴
    # 技巧：伪造一天的数据来提取周期曲线
    dummy_date = '2025-01-01' # 任意一天
    future_day = pd.date_range(start=f'{dummy_date} 00:00:00', end=f'{dummy_date} 23:55:00', freq='5min')
    df_dummy = pd.DataFrame({'ds': future_day})
    
    # 预测并提取 'daily' 组件 (我们之前手动命名的 seasonality)
    # 注意：Prophet 的 add_seasonality 会把结果存在列名 'daily' 中
    fc_daily_work = m_work.predict(df_dummy)['daily']
    fc_daily_rest = m_rest.predict(df_dummy)['daily']
    
    # 绘图
    plt.figure(figsize=(10, 6))
    hours = future_day.hour + future_day.minute / 60.0
    
    plt.plot(hours, fc_daily_work, label='Workday Profile (High Traffic)', color='#d62728', lw=2.5) # Red
    plt.plot(hours, fc_daily_rest, label='Weekend Profile (Low Traffic)', color='#2ca02c', lw=2.5)  # Green
    
    plt.title('Component 2: Daily Seasonality (Workday vs Weekend)', fontsize=14)
    plt.xlabel('Hour of Day')
    plt.ylabel('Seasonality Impact')
    plt.xticks(np.arange(0, 25, 2)) # 每2小时一个刻度
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path_daily = os.path.join(IMAGE_DIR, 'component_2_daily.png')
    plt.savefig(save_path_daily, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path_daily}")

    # ==========================================
    # 3. Weekly Component (周效应重构)
    # ==========================================
    print("Generating Weekly Component Plot...")
    
    # 由于我们拆分了模型，Weekly 效应体现为“周中”和“周末”的总量差异
    # 我们统计测试集中每一天的平均预测流量来展示这个效应
    
    # 使用全量预测结果
    yhat_combined = np.where(
        df['is_weekend'], 
        forecast_rest['yhat'], 
        forecast_work['yhat']
    )
    df['yhat'] = yhat_combined
    df['day_of_week'] = df['ds'].dt.dayofweek # 0=Mon, 6=Sun
    
    # 计算每天的平均流量
    weekly_profile = df.groupby('day_of_week')['yhat'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(days, weekly_profile, color=['#1f77b4']*5 + ['#ff7f0e']*2, alpha=0.8)
    
    # 标注数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')

    plt.title('Component 3: Effective Weekly Seasonality', fontsize=14)
    plt.xlabel('Day of Week')
    plt.ylabel('Average Predicted Traffic')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加图例说明
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1f77b4', label='Workday Mode'),
                       Patch(facecolor='#ff7f0e', label='Weekend Mode')]
    plt.legend(handles=legend_elements)
    
    save_path_weekly = os.path.join(IMAGE_DIR, 'component_3_weekly.png')
    plt.savefig(save_path_weekly, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path_weekly}")
    
    print("All components visualized successfully.")

if __name__ == "__main__":
    visualize_components()