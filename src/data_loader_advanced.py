import pandas as pd
import numpy as np
import os
from pathlib import Path

# ============================
# 1. 路径配置
# ============================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
RAW_PATH = project_root / "data" / "raw"
PROCESSED_PATH = project_root / "data" / "processed"

def create_master_table():
    print("🚀 Building the Ultimate Master Table (Integration of 6 Files)...")
    
    # --- Check Files ---
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"❌ Data directory not found: {RAW_PATH}")

    # --- 1. Load All Raw Data ---
    print("   Loading raw CSVs...")
    df_hall = pd.read_csv(RAW_PATH / 'hall_calls.csv')
    df_car = pd.read_csv(RAW_PATH / 'car_calls.csv')
    df_stops = pd.read_csv(RAW_PATH / 'car_stops.csv')
    df_load = pd.read_csv(RAW_PATH / 'load_changes.csv')
    df_maint = pd.read_csv(RAW_PATH / 'maintenance_mode.csv')
    
    # Time Conversion (Standardization)
    for df in [df_hall, df_car, df_stops, df_load, df_maint]:
        df['Time'] = pd.to_datetime(df['Time'])

    # --- 2. Aggregation (Resampling to 5min) ---
    # 我们以 5分钟 为一个“原子时间片”
    freq = '5min'
    
    print("   Aggregating features...")
    
    # [A] Hall Calls: Split Up vs Down
    # 这对区分早晚高峰至关重要
    hall_up = df_hall[df_hall['Direction'] == 'Up'].set_index('Time').resample(freq)['Floor'].count().rename('hall_up')
    hall_down = df_hall[df_hall['Direction'] == 'Down'].set_index('Time').resample(freq)['Floor'].count().rename('hall_down')
    
    # [B] Car Calls: Internal Demand
    # Only count 'Register' actions (Ignore Cancel)
    car_calls = df_car[df_car['Action'] == 'Register'].set_index('Time').resample(freq)['Floor'].count().rename('car_calls')
    
    # [C] Load Changes: Passenger Flow
    # Convert kg to people (75kg/person)
    df_load['people_in'] = (df_load['Load In (kg)'] / 75).round()
    df_load['people_out'] = (df_load['Load Out (kg)'] / 75).round()
    
    load_in = df_load.set_index('Time').resample(freq)['people_in'].sum().rename('people_in')
    load_out = df_load.set_index('Time').resample(freq)['people_out'].sum().rename('people_out')
    
    # [D] Car Stops: Service Intensity
    stops = df_stops.set_index('Time').resample(freq)['Floor'].count().rename('total_stops')
    
    # [E] Maintenance: Disruptions
    # Count distinct maintenance entries in this window
    maint_events = df_maint[df_maint['Action'] == 'Enter'].set_index('Time').resample(freq)['Elevator ID'].count().rename('maint_events')

    # --- 3. Merge All Features ---
    # concat 会自动按时间索引对齐 (Outer Join)
    dfs = [hall_up, hall_down, car_calls, load_in, load_out, stops, maint_events]
    df_master = pd.concat(dfs, axis=1).fillna(0)
    
    # --- 4. Enhance Time Features ---
    # 添加你要求的“星期几”等信息
    df_master['hour'] = df_master.index.hour
    df_master['minute'] = df_master.index.minute
    df_master['day_of_week'] = df_master.index.dayofweek # 0=Mon, 6=Sun
    df_master['day_name'] = df_master.index.day_name()
    df_master['is_weekend'] = df_master['day_of_week'] >= 5
    
    # Add Total Demand
    df_master['total_demand'] = df_master['hall_up'] + df_master['hall_down'] + df_master['car_calls']
    
    # --- 5. Save ---
    PROCESSED_PATH.mkdir(exist_ok=True)
    save_path = PROCESSED_PATH / 'master_traffic_table.csv'
    
    print(f"✅ Master Table Created! Shape: {df_master.shape}")
    print(f"💾 Saved to: {save_path}")
    print("\n🔍 Preview (First 5 rows):")
    print(df_master[['day_name', 'hour', 'hall_up', 'hall_down', 'people_in', 'total_stops']].head())
    
    return df_master

if __name__ == "__main__":
    create_master_table()