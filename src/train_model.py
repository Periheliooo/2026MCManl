import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_and_evaluate():
    # 1. 加载数据并恢复时间信息以进行分割
    # 注意：实际训练时会移除时间列
    if os.path.exists('master_traffic_table.csv'):
        df = pd.read_csv('master_traffic_table.csv')
    else:
        df = pd.read_csv('data/processed/master_traffic_table.csv')
        
    df['Time'] = pd.to_datetime(df['Time'])
    
    # 辅助特征生成标签 (与之前逻辑一致)
    df['hour_minute'] = df['Time'].dt.hour + df['Time'].dt.minute / 60.0
    
    def get_traffic_state(row):
        if row['is_weekend']: return 1 # Idle
        hm = row['hour_minute']
        if 9.5 <= hm < 11.0: return 2 # Morning
        elif 11.5 <= hm < 14.5: return 3 # Lunch
        elif 18.0 <= hm < 20.0: return 4 # Evening
        else: return 1 # Idle

    df['traffic_state_id'] = df.apply(get_traffic_state, axis=1)
    
    # 2. 按日期分割训练集和测试集
    unique_dates = sorted(df['Time'].dt.date.unique())
    train_dates = unique_dates[:20]  # 前20天
    test_dates = unique_dates[-10:]  # 后10天
    
    train_df = df[df['Time'].dt.date.isin(train_dates)]
    test_df = df[df['Time'].dt.date.isin(test_dates)]
    
    # 3. 准备特征 (移除时间列)
    feature_cols = ['hall_up', 'hall_down', 'car_calls', 'people_in', 'people_out', 
                   'total_stops', 'maint_events', 'total_demand']
    target_col = 'traffic_state_id'
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 4. 使用最优参数训练模型
    # Best Params found: n_estimators=100, min_samples_leaf=2
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=42)
    rf.fit(X_train, y_train)
    
    # 5. 验证
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n最终模型准确率: {accuracy:.4f}")
    print("\n详细分类报告:")
    target_names = ['Idle', 'Morning', 'Lunch', 'Evening']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 6. 保存模型
    os.makedirs('models', exist_ok=True)
    model_path = 'models/traffic_classifier_rf.joblib'
    joblib.dump(rf, model_path)
    print(f"模型已保存至: {model_path}")

if __name__ == "__main__":
    train_and_evaluate()