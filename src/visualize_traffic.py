import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def visualize_traffic():
    # ==============================
    # 1. 路径配置 (Path Configuration)
    # ==============================
    # 获取当前脚本所在目录 (src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录 (src的上一级)
    project_root = os.path.dirname(current_dir)
    
    # 定义数据输入路径和图片输出路径
    data_path = os.path.join(project_root, 'data', 'processed', 'master_traffic_table.csv')
    img_dir = os.path.join(project_root, 'images')
    
    # 确保图片输出目录存在
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        print(f"Created directory: {img_dir}")

    # ==============================
    # 2. 数据加载与预处理 (Data Loading)
    # ==============================
    print(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
        print("Error: Data file not found. Please check the path.")
        return

    df = pd.read_csv(data_path)
    
    # 转换时间列
    df['Time'] = pd.to_datetime(df['Time'])
    
    # 提取时间特征用于绘图 (如果CSV中没有这些列)
    if 'hour' not in df.columns:
        df['hour'] = df['Time'].dt.hour
    if 'minute' not in df.columns:
        df['minute'] = df['Time'].dt.minute
    
    # 计算一天中的浮点时间 (例如 8.5 代表 8:30)
    df['TimeOfDay'] = df['hour'] + df['minute'] / 60.0

    # 区分工作日数据
    # 假设 'is_weekend' 列存在，如果不存在则手动创建
    if 'is_weekend' not in df.columns:
        df['day_of_week'] = df['Time'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'] >= 5
    
    df_weekday = df[~df['is_weekend']].copy()

    # ==============================
    # 3. 可视化 1: 工作日平均流量模式
    #    (Weekday Average Traffic Pattern)
    # ==============================
    print("Generating Weekday Traffic Pattern plot...")
    
    # 按时间段聚合平均数据
    daily_pattern = df_weekday.groupby(['hour', 'minute']).agg({
        'hall_up': 'mean',
        'hall_down': 'mean',
        'total_demand': 'mean'
    }).reset_index()
    
    daily_pattern['TimeOfDay'] = daily_pattern['hour'] + daily_pattern['minute'] / 60.0

    plt.figure(figsize=(14, 7))
    
    # 绘制总需求背景阴影
    plt.fill_between(daily_pattern['TimeOfDay'], 0, daily_pattern['total_demand'], 
                     color='gray', alpha=0.1, label='Total Demand Area')
    
    # 绘制各指标曲线
    plt.plot(daily_pattern['TimeOfDay'], daily_pattern['hall_up'], 
             label='Hall Up (Avg)', color='#2ca02c', linewidth=2)  # 绿色代表上行
    plt.plot(daily_pattern['TimeOfDay'], daily_pattern['hall_down'], 
             label='Hall Down (Avg)', color='#d62728', linewidth=2) # 红色代表下行
    plt.plot(daily_pattern['TimeOfDay'], daily_pattern['total_demand'], 
             label='Total Demand', color='#1f77b4', linestyle='--', alpha=0.7) # 蓝色虚线总需求

    # 图表装饰
    plt.title('Average Elevator Traffic Patterns (Typical Weekday)', fontsize=16)
    plt.xlabel('Hour of Day (0-24)', fontsize=12)
    plt.ylabel('Average Calls / Demand (per 5 min)', fontsize=12)
    plt.xticks(np.arange(0, 25, 1))
    plt.xlim(0, 24)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', fontsize=11)
    
    # 添加文字标注 (基于典型电梯模式)
    plt.text(8.5, daily_pattern['hall_up'].max()*0.8, 'Morning Up-Peak', color='#2ca02c', fontweight='bold')
    plt.text(17.5, daily_pattern['hall_down'].max()*0.8, 'Evening Down-Peak', color='#d62728', fontweight='bold')
    plt.text(12.5, daily_pattern['total_demand'].max()*0.9, 'Lunch Hour', color='#1f77b4', fontweight='bold', ha='center')

    # 保存图片
    output_file_1 = os.path.join(img_dir, 'weekday_traffic_pattern.png')
    plt.savefig(output_file_1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file_1}")

    # ==============================
    # 4. 可视化 2: 每周流量热力图
    #    (Weekly Demand Heatmap)
    # ==============================
    print("Generating Weekly Heatmap...")

    # 准备透视表数据：行是星期，列是小时，值是平均总需求
    heatmap_data = df.groupby(['day_name', 'hour'])['total_demand'].mean().reset_index()
    
    # 确保星期顺序正确
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data['day_name'] = pd.Categorical(heatmap_data['day_name'], categories=days_order, ordered=True)
    
    pivot_table = heatmap_data.pivot(index='day_name', columns='hour', values='total_demand')

    plt.figure(figsize=(16, 6))
    sns.heatmap(pivot_table, cmap='YlOrRd', linewidths=.5, cbar_kws={'label': 'Avg Total Demand'})
    
    plt.title('Heatmap of Elevator Demand by Day and Hour', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Day of Week', fontsize=12)
    plt.xticks(rotation=0)
    
    # 保存图片
    output_file_2 = os.path.join(img_dir, 'weekly_demand_heatmap.png')
    plt.savefig(output_file_2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file_2}")

    print("All visualizations completed successfully.")

if __name__ == "__main__":
    visualize_traffic()