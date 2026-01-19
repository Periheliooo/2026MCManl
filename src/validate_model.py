# src/validate_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

# 导入高级模型和数据处理函数
from predictor import create_advanced_model, prepare_data 

# --- 路径配置 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'traffic_5min.csv')
IMAGE_DIR = os.path.join(PROJECT_ROOT, 'images')

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def validate():
    # 1. 数据读取
    print(f"Reading data from: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH)
    
    # [关键步骤] 使用 predictor 中定义的特征工程处理数据
    # 这会自动添加 'is_workday' 和 'is_weekend' 列
    df = prepare_data(df_raw)
    df = df.sort_values('ds').reset_index(drop=True)
    
    # 2. 划分训练/测试集
    points_per_day = 288
    test_days = 5
    test_size = points_per_day * test_days
    
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    
    print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # 3. 模型训练
    print("Training ADVANCED model...")
    model = create_advanced_model()
    model.fit(train_df)
    
    # 4. 预测
    print("Generating forecast...")
    future = model.make_future_dataframe(periods=test_size, freq='5min')
    
    # [关键步骤] 为未来时间点补充同样的特征
    # 必须手动重新计算 is_weekend/is_workday，因为 make_future_dataframe 只生成日期
    future['is_weekend'] = future['ds'].dt.dayofweek >= 5
    future['is_workday'] = ~future['is_weekend']
    
    forecast = model.predict(future)
    
    # 提取预测值
    y_pred = forecast['yhat'].iloc[-test_size:].values
    y_pred[y_pred < 0] = 0 # 物理约束：人数不为负
    
    y_true = test_df['y'].values
    dates_test = test_df['ds'].values
    
    # 5. 评估指标
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"Validation Results: R2={r2:.4f}, RMSE={rmse:.4f}")
    
    # --- 6. 全套可视化 (4张图) ---
    print("Generating full suite of plots...")

    # 图 1: 全局概览 (最后 5 天)
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_true, label='Observed', color='black', alpha=0.5)
    plt.plot(dates_test, y_pred, label='Predicted (Advanced)', color='#0072B2', lw=1.5)
    plt.title(f'1. Forecast Overview (All 5 Test Days) - RMSE: {rmse:.2f}', fontsize=14)
    plt.ylabel('People In')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path_1 = os.path.join(IMAGE_DIR, 'val_1_overview.png')
    plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path_1}")

    # 图 2: 散点回归
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

    # 图 3: 残差分布
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

    # 图 4: 局部细节 (重点关注最后 3 天)
    # 包含 周五(工作日) -> 周六/日(周末) 的切换
    plt.figure(figsize=(14, 7))
    zoom_days = 3 
    zoom_points = 288 * zoom_days
    
    # 为了防止索引越界 (如果测试集不足3天)
    safe_zoom_points = min(zoom_points, len(dates_test))
    
    plt.plot(dates_test[-safe_zoom_points:], y_true[-safe_zoom_points:], 'k.', alpha=0.3, label='Observed')
    plt.plot(dates_test[-safe_zoom_points:], y_pred[-safe_zoom_points:], 'r-', lw=2, label='Predicted (Advanced)')
    
    plt.title(f'4. Detail View: Weekday vs Weekend Transition (Last {zoom_days} Days)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    
    save_path_4 = os.path.join(IMAGE_DIR, 'val_4_zoomin.png')
    plt.savefig(save_path_4, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path_4}")

if __name__ == "__main__":
    validate()