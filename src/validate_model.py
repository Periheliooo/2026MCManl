# src/validate_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

# 从同目录导入模型工厂
from predictor import create_standard_model 

# --- 路径配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'traffic_5min.csv')
IMAGE_DIR = os.path.join(PROJECT_ROOT, 'images')

# 确保图片目录存在
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def validate():
    # 1. 数据读取
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件未找到: {DATA_PATH}")
        
    print(f"Reading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df['ds'] = pd.to_datetime(df['Time'])
    df['y'] = df['people_in']
    df = df.sort_values('ds').reset_index(drop=True)
    
    # 2. 划分训练/测试集 (Hold-out Validation)
    points_per_day = 288
    test_days = 5
    test_size = points_per_day * test_days
    
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    
    print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # 3. 模型训练
    print("Training validation model...")
    model = create_standard_model()
    model.fit(train_df)
    
    # 4. 预测
    future = model.make_future_dataframe(periods=test_size, freq='5min')
    forecast = model.predict(future)
    
    y_pred = forecast['yhat'].iloc[-test_size:].values
    y_pred[y_pred < 0] = 0 # 物理约束
    
    y_true = test_df['y'].values
    dates_test = test_df['ds'].values
    
    # 5. 评估指标
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"Validation Results: R2={r2:.4f}, RMSE={rmse:.4f}")
    
    # --- 6. 可视化 (改为分别保存4张图) ---
    print("Generating separate plots...")

    # 图 1: 时序概览 (Forecast Overview)
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_true, label='Observed (Truth)', color='black', alpha=0.5)
    plt.plot(dates_test, y_pred, label='Predicted (Model)', color='#0072B2', lw=1.5)
    plt.title(f'Forecast Overview (Last 5 Days) - RMSE: {rmse:.2f}', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('People In')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path_1 = os.path.join(IMAGE_DIR, 'val_1_overview.png')
    plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
    plt.close() # 关闭画布，释放内存
    print(f"Saved: {save_path_1}")

    # 图 2: 散点回归 (Obs vs Pred)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.1, color='purple')
    limit = max(y_true.max(), y_pred.max())
    plt.plot([0, limit], [0, limit], 'k--', alpha=0.7, label='Perfect Fit') # y=x 线
    plt.title(f'Observed vs Predicted ($R^2={r2:.3f}$)', fontsize=14)
    plt.xlabel('Observed Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path_2 = os.path.join(IMAGE_DIR, 'val_2_scatter.png')
    plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path_2}")

    # 图 3: 残差分布 (Residuals)
    plt.figure(figsize=(10, 6))
    residuals = y_true - y_pred
    plt.hist(residuals, bins=30, density=True, color='gray', alpha=0.7, edgecolor='black')
    plt.title('Residual Distribution (Normality Check)', fontsize=14)
    plt.xlabel('Residual (Error)')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    save_path_3 = os.path.join(IMAGE_DIR, 'val_3_residuals.png')
    plt.savefig(save_path_3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path_3}")

    # 图 4: 局部细节 (Zoom-in Last 24h)
    plt.figure(figsize=(12, 6))
    zoom_points = 288 # Last 24h
    plt.plot(dates_test[-zoom_points:], y_true[-zoom_points:], 'k.-', alpha=0.4, label='Observed')
    plt.plot(dates_test[-zoom_points:], y_pred[-zoom_points:], 'b.-', lw=2, label='Predicted')
    plt.title('Detail View: Last 24 Hours', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('People In')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 优化X轴时间显示 (防止重叠)
    plt.gcf().autofmt_xdate()
    
    save_path_4 = os.path.join(IMAGE_DIR, 'val_4_zoomin.png')
    plt.savefig(save_path_4, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path_4}")

if __name__ == "__main__":
    validate()