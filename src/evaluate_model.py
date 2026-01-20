import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

def evaluate_model():
    # 1. 加载数据并恢复时间信息 (用于分割测试集)
    if os.path.exists('master_traffic_table.csv'):
        df = pd.read_csv('master_traffic_table.csv')
    else:
        # Fallback path
        df = pd.read_csv('data/processed/master_traffic_table.csv')
        
    df['Time'] = pd.to_datetime(df['Time'])
    
    # 重新生成标签 (保持逻辑一致)
    df['hour_minute'] = df['Time'].dt.hour + df['Time'].dt.minute / 60.0
    
    def get_traffic_state(row):
        if row['is_weekend']: return 1 # Idle
        hm = row['hour_minute']
        if 9.5 <= hm < 11.0: return 2 # Morning
        elif 11.5 <= hm < 14.5: return 3 # Lunch
        elif 18.0 <= hm < 20.0: return 4 # Evening
        else: return 1 # Idle

    df['traffic_state_id'] = df.apply(get_traffic_state, axis=1)
    
    # 2. 分割训练集和测试集 (前20天训练，后10天测试)
    unique_dates = sorted(df['Time'].dt.date.unique())
    train_dates = unique_dates[:20]
    test_dates = unique_dates[-10:]
    
    train_df = df[df['Time'].dt.date.isin(train_dates)]
    test_df = df[df['Time'].dt.date.isin(test_dates)]
    
    feature_cols = ['hall_up', 'hall_down', 'car_calls', 'people_in', 'people_out', 
                   'total_stops', 'maint_events', 'total_demand']
    target_col = 'traffic_state_id'
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print(f"Testing on {len(X_test)} samples from {test_dates[0]} to {test_dates[-1]}")
    
    # 3. 训练模型 (使用之前确定的最优参数)
    print("Training Random Forest Model...")
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=42)
    rf.fit(X_train, y_train)
    
    # 4. 预测
    y_pred = rf.predict(X_test)
    
    # 5. 计算相关系数
    eval_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    correlation = eval_df.corr().iloc[0, 1]
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n--- Evaluation Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Correlation Coefficient: {correlation:.4f}")
    
    # 6. 绘制散点图 (Actual vs Predicted)
    plt.figure(figsize=(10, 6))
    
    # 添加抖动 (Jitter) 以防点重叠
    jitter_x = np.random.normal(0, 0.1, size=len(eval_df))
    jitter_y = np.random.normal(0, 0.1, size=len(eval_df))
    
    plt.scatter(eval_df['Actual'] + jitter_x, eval_df['Predicted'] + jitter_y, 
                alpha=0.3, c='blue', s=30)
    
    # 添加对角参考线
    plt.plot([1, 4], [1, 4], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.title(f'Actual vs Predicted Traffic State\nCorrelation: {correlation:.4f}', fontsize=14)
    plt.xlabel('Actual State ID', fontsize=12)
    plt.ylabel('Predicted State ID', fontsize=12)
    plt.xticks([1, 2, 3, 4], ['Idle(1)', 'Morning(2)', 'Lunch(3)', 'Evening(4)'])
    plt.yticks([1, 2, 3, 4], ['Idle(1)', 'Morning(2)', 'Lunch(3)', 'Evening(4)'])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_file = 'evaluation_scatter_plot.png'
    plt.savefig(output_file)
    print(f"\nScatter plot saved to: {output_file}")
    # plt.show() # 如果在本地运行可以取消注释

if __name__ == "__main__":
    evaluate_model()