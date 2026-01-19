# src/validate_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

# 导入双模型工厂函数
from predictor import create_single_mode_model, prepare_data

# --- 路径配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'traffic_5min.csv')
IMAGE_DIR = os.path.join(PROJECT_ROOT, 'images')

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def validate():
    # 1. 数据准备
    print(f"Reading data from: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH)
    df = prepare_data(df_raw) # 自动打上 is_weekend 标签
    df = df.sort_values('ds').reset_index(drop=True)
    
    # 2. 划分训练/测试集
    points_per_day = 288
    test_days = 5
    test_size = points_per_day * test_days
    
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    
    print(f"Total Train: {len(train_df)}, Total Test: {len(test_df)}")
    
    # --- 3. 双模型分别训练 ---
    
    # A. 拆分训练数据
    train_work = train_df[~train_df['is_weekend']].copy()
    train_rest = train_df[train_df['is_weekend']].copy()
    
    # B. 训练
    print("Training Workday Model...")
    m_work = create_single_mode_model(is_workday_mode=True)
    m_work.fit(train_work)
    
    print("Training Weekend Model...")
    m_rest = create_single_mode_model(is_workday_mode=False)
    m_rest.fit(train_rest)
    
    # --- 4. 双模型分别预测 ---
    
    # A. 准备预测时间表
    future_dates = pd.DataFrame({'ds': test_df['ds']})
    future_dates['is_weekend'] = future_dates['ds'].dt.dayofweek >= 5
    
    # B. 拆分预测请求
    future_work = future_dates[~future_dates['is_weekend']].copy()
    future_rest = future_dates[future_dates['is_weekend']].copy()
    
    # C. 分别预测并合并
    pred_work = pd.DataFrame()
    if len(future_work) > 0:
        fc_work = m_work.predict(future_work)
        pred_work = fc_work[['ds', 'yhat']].copy()
        
    pred_rest = pd.DataFrame()
    if len(future_rest) > 0:
        fc_rest = m_rest.predict(future_rest)
        pred_rest = fc_rest[['ds', 'yhat']].copy()
    
    # 合并并排序，确保时间轴连续
    pred_combined = pd.concat([pred_work, pred_rest]).sort_values('ds')
    
    # 提取最终预测值
    y_pred = pred_combined['yhat'].values
    y_pred[y_pred < 0] = 0 # 物理约束
    
    y_true = test_df['y'].values
    dates_test = test_df['ds'].values
    
    # 5. 评估指标
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"Validation Results: R2={r2:.4f}, RMSE={rmse:.4f}")
    
    # --- 6. 全套可视化 (4张图) ---
    print("Generating full suite of plots...")

    # 图 1: 全局概览 (Forecast Overview)
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_true, label='Observed', color='black', alpha=0.5)
    plt.plot(dates_test, y_pred, label='Predicted (Split-Model)', color='#0072B2', lw=1.5)
    plt.title(f'1. Forecast Overview (RMSE: {rmse:.2f})', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('People In')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path_1 = os.path.join(IMAGE_DIR, 'val_1_overview.png')
    plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path_1}")

    # 图 2: 散点回归 (Obs vs Pred)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.1, color='purple')
    limit = max(y_true.max(), y_pred.max())
    plt.plot([0, limit], [0, limit], 'k--', alpha=0.7)
    plt.title(f'2. Observed vs Predicted ($R^2={r2:.3f}$)', fontsize=14)
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.grid(True, alpha=0.3)
    save_path_2 = os.path.join(IMAGE_DIR, 'val_2_scatter.png')
    plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path_2}")

    # 图 3: 残差分布 (Residuals)
    plt.figure(figsize=(10, 6))
    residuals = y_true - y_pred
    plt.hist(residuals, bins=50, density=True, color='gray', alpha=0.7, edgecolor='black')
    plt.title('3. Residual Distribution', fontsize=14)
    plt.xlabel('Error (True - Pred)')
    plt.grid(True, alpha=0.3)
    save_path_3 = os.path.join(IMAGE_DIR, 'val_3_residuals.png')
    plt.savefig(save_path_3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path_3}")

    # 图 4: 局部细节 (Zoom-in Last 3 Days)
    # 专门展示 Workday -> Weekend 的无缝切换
    plt.figure(figsize=(14, 7))
    zoom_days = 3
    # 确保不越界
    start_idx = max(0, len(dates_test) - 288 * zoom_days)
    
    plt.plot(dates_test[start_idx:], y_true[start_idx:], 'k.', alpha=0.3, label='Observed')
    plt.plot(dates_test[start_idx:], y_pred[start_idx:], 'g-', lw=2, label='Predicted (Split-Model)')
    
    plt.title(f'4. Detail View: Workday/Weekend Transition (Last {zoom_days} Days)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    
    save_path_4 = os.path.join(IMAGE_DIR, 'val_4_zoomin.png')
    plt.savefig(save_path_4, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path_4}")

if __name__ == "__main__":
    validate()