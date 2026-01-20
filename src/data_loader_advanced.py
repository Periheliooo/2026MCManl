import pandas as pd
import os

def create_labeled_training_data():
    input_path = 'data/processed/master_traffic_table.csv'
    output_path = 'data/processed/labeled_traffic_data_for_training.csv'
    
    # 1. 读取数据
    if not os.path.exists(input_path):
        # Fallback for demonstration
        input_path = 'master_traffic_table.csv'
        
    df = pd.read_csv(input_path)
    df['Time'] = pd.to_datetime(df['Time'])
    
    # 辅助计算列
    df['hour_minute'] = df['Time'].dt.hour + df['Time'].dt.minute / 60.0
    
    # 2. 定义修正后的判定逻辑
    def get_traffic_state(row):
        # 规则 1: 周末全天为空闲期
        if row['is_weekend']:
            return 1  # Idle Period
        
        # 规则 2: 工作日根据时间段判定
        hm = row['hour_minute']
        
        if 9.5 <= hm < 11.0:
            return 2  # Morning Peak (09:30 - 11:00)
        elif 11.5 <= hm < 14.5:
            return 3  # Lunch Peak (11:30 - 14:30)
        elif 18.0 <= hm < 20.0:
            return 4  # Evening Peak (18:00 - 20:00)
        else:
            return 1  # Idle Period (Other times)

    # 应用逻辑
    df['traffic_state_id'] = df.apply(get_traffic_state, axis=1)
    
    # 添加文本标签方便查看 (训练时可选择是否保留)
    state_map = {
        1: 'Idle Period',
        2: 'Morning Peak',
        3: 'Lunch Peak',
        4: 'Evening Peak'
    }
    df['traffic_state'] = df['traffic_state_id'].map(state_map)
    
    # 3. 数据清洗：移除时间列
    # 根据要求，去除 Time, hour, day_of_week 等，只保留流量特征和标签
    # 这样模型只能看到：hall_up, hall_down, car_calls, people_in, people_out, total_stops, maint_events, total_demand
    cols_to_drop = ['Time', 'hour', 'minute', 'day_of_week', 'day_name', 'is_weekend', 'hour_minute']
    training_df = df.drop(columns=cols_to_drop)
    
    # 4. 保存
    training_df.to_csv(output_path, index=False)
    
    print(f"处理完成。训练数据已保存至: {output_path}")
    print("保留的特征列:", training_df.columns.tolist())
    print("\n各状态样本数分布:")
    print(training_df['traffic_state'].value_counts())

if __name__ == "__main__":
    create_labeled_training_data()