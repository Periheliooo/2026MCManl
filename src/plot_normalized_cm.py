import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import os

def plot_normalized_cm():
    # 1. 准备数据 (同前)
    if os.path.exists('master_traffic_table.csv'):
        df = pd.read_csv('master_traffic_table.csv')
    else:
        df = pd.read_csv('data/processed/master_traffic_table.csv')
        
    df['Time'] = pd.to_datetime(df['Time'])
    df['hour_minute'] = df['Time'].dt.hour + df['Time'].dt.minute / 60.0
    
    def get_traffic_state(row):
        if row['is_weekend']: return 1 
        hm = row['hour_minute']
        if 9.5 <= hm < 11.0: return 2
        elif 11.5 <= hm < 14.5: return 3
        elif 18.0 <= hm < 20.0: return 4
        else: return 1

    df['traffic_state_id'] = df.apply(get_traffic_state, axis=1)
    
    unique_dates = sorted(df['Time'].dt.date.unique())
    train_dates = unique_dates[:20]
    test_dates = unique_dates[-10:]
    
    train_df = df[df['Time'].dt.date.isin(train_dates)]
    test_df = df[df['Time'].dt.date.isin(test_dates)]
    
    cols = ['hall_up', 'hall_down', 'car_calls', 'people_in', 'people_out', 
            'total_stops', 'maint_events', 'total_demand']
    
    X_train = train_df[cols]
    y_train = train_df['traffic_state_id']
    X_test = test_df[cols]
    y_test = test_df['traffic_state_id']
    
    # 2. 训练
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    # 3. 计算归一化矩阵
    cm = confusion_matrix(y_test, y_pred)
    # 核心步骤: 按行求和并相除，得到百分比
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 4. 绘图
    labels = ['Idle', 'Morning', 'Lunch', 'Evening']
    plt.figure(figsize=(10, 8))
    
    # fmt='.1%' 让数值显示为百分比 (如 85.2%)
    sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
    
    plt.title('Normalized Confusion Matrix (Recall)', fontsize=16)
    plt.ylabel('Actual State', fontsize=12)
    plt.xlabel('Predicted State', fontsize=12)
    
    output_path = 'confusion_matrix_normalized.png'
    plt.savefig(output_path)
    print(f"归一化混淆矩阵已保存至: {output_path}")

if __name__ == "__main__":
    plot_normalized_cm()