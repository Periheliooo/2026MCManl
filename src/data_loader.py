import pandas as pd
import numpy as np
import os
from pathlib import Path

# ==========================================
# 1. 路径配置 (使用 pathlib 解决路径报错)
# ==========================================
# 定位到 src/data_loader.py 所在的文件夹
current_dir = Path(__file__).resolve().parent
# 回溯到项目根目录 (MCM_Project/)
project_root = current_dir.parent
# 定义数据目录
RAW_PATH = project_root / "data" / "raw"
PROCESSED_PATH = project_root / "data" / "processed"

def load_and_clean_data():
    """读取原始 csv 并返回清洗后的 DataFrame"""
    print(f"📂 Reading data from: {RAW_PATH}")
    
    # --- 1. Load Hall Calls ---
    hall_path = RAW_PATH / 'hall_calls.csv'
    if not hall_path.exists():
        raise FileNotFoundError(f"❌ 文件缺失: {hall_path}")
        
    df_hall = pd.read_csv(hall_path)
    df_hall['Time'] = pd.to_datetime(df_hall['Time'])
    df_hall = df_hall.dropna(subset=['Floor'])
    df_hall['source_type'] = 'hall_call'

    # --- 2. Load Load Changes ---
    load_path = RAW_PATH / 'load_changes.csv'
    df_load = pd.read_csv(load_path)
    df_load['Time'] = pd.to_datetime(df_load['Time'])
    # 75kg/人 估算
    df_load['people_in'] = (df_load['Load In (kg)'] / 75).round().astype(int)
    df_load['source_type'] = 'load_change'

    # --- 3. Aggregate Traffic (5min) ---
    print("📊 Processing Traffic Table...")
    traffic_hall = df_hall.set_index('Time').resample('5min')['Floor'].count().rename('hall_call_count')
    traffic_load = df_load.set_index('Time').resample('5min')[['people_in']].sum()
    
    traffic_5min = pd.concat([traffic_hall, traffic_load], axis=1).fillna(0)
    
    # --- 4. Merge Raw Events ---
    print("🔗 Processing Event Log...")
    # 简化版合并，仅演示
    cols = ['Time', 'Elevator ID', 'Floor', 'source_type']
    raw_events = pd.concat([
        df_hall[cols], 
        df_load[cols]
    ]).sort_values('Time').reset_index(drop=True)

    return traffic_5min, raw_events

if __name__ == "__main__":
    # ==========================================
    # 🧪 测试与保存 (直接运行此脚本即可生成文件)
    # ==========================================
    
    # 1. 执行清洗
    traffic, events = load_and_clean_data()
    
    # 2. 确保 processed 文件夹存在 (如果不存在，自动创建)
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    
    # 3. 保存文件 (持久化)
    traffic_file = PROCESSED_PATH / 'traffic_5min.csv'
    events_file = PROCESSED_PATH / 'events_log.csv'
    
    print(f"💾 Saving to {PROCESSED_PATH}...")
    traffic.to_csv(traffic_file)
    events.to_csv(events_file, index=False) # event log 不需要索引列
    
    print("✅ Success! Files generated:")
    print(f"   - {traffic_file}")
    print(f"   - {events_file}")