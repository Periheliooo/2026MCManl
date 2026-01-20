import pandas as pd
import os

def label_traffic_states():
    # 定义文件路径
    input_path = 'data/processed/master_traffic_table.csv'
    output_path = 'data/processed/master_traffic_table_with_states.csv'
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. 读取数据
    if not os.path.exists(input_path):
        # 如果找不到指定路径，尝试在当前目录查找（用于演示）
        if os.path.exists('master_traffic_table.csv'):
            input_path = 'master_traffic_table.csv'
        else:
            raise FileNotFoundError(f"Cannot find {input_path}")
            
    df = pd.read_csv(input_path)
    df['Time'] = pd.to_datetime(df['Time'])
    
    # 计算当天的小时数值（例如 9.5 代表 9:30）
    # 这一步是为了方便进行时间窗口的比较
    df['hour_minute'] = df['Time'].dt.hour + df['Time'].dt.minute / 60.0

    # 2. 定义判定函数 (基于分析得出的时间窗口)
    def get_state(row):
        hm = row['hour_minute']
        
        # 早高峰: 09:30 - 11:00
        if 9.5 <= hm < 11.0:
            return 2  # 早高峰 ID
        
        # 午高峰: 11:30 - 14:30
        elif 11.5 <= hm < 14.5:
            return 3  # 午高峰 ID
        
        # 晚高峰: 18:00 - 20:00
        elif 18.0 <= hm < 20.0:
            return 4  # 晚高峰 ID
        
        # 空闲期: 其他所有时间
        else:
            return 1  # 空闲期 ID

    # 3. 应用判定逻辑
    df['traffic_state_id'] = df.apply(get_state, axis=1)
    
    # 添加可读的文本标签映射
    state_map = {
        1: 'Idle Period',
        2: 'Morning Peak',
        3: 'Lunch Peak',
        4: 'Evening Peak'
    }
    df['traffic_state'] = df['traffic_state_id'].map(state_map)
    
    # 4. 保存结果
    # 移除辅助计算列
    df_final = df.drop(columns=['hour_minute'])
    df_final.to_csv(output_path, index=False)
    
    print(f"处理完成。已标记 {len(df)} 条数据。")
    print(f"数据已保存至: {output_path}")
    print("\n各状态数据分布:")
    print(df['traffic_state'].value_counts())

if __name__ == "__main__":
    label_traffic_states()