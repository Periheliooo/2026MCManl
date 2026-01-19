import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# 设置路径 (保持你原有的结构)
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
PROCESSED_PATH = project_root / "data" / "processed"
IMG_PATH = project_root / "images"

def plot_traffic_heatmap():
    # 1. 读取数据
    # 请确保该路径下有你的 traffic_5min.csv 文件
    csv_path = PROCESSED_PATH / 'traffic_5min.csv'
    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df['Time'] = pd.to_datetime(df['Time'])
    
    # 2. 特征提取
    df['hour'] = df['Time'].dt.hour
    df['date'] = df['Time'].dt.date
    df['minute'] = df['Time'].dt.minute
    df['time_float'] = df['hour'] + df['minute']/60
    
    # 3. 构造透视表
    df['total_traffic'] = df['hall_call_count'] + df['people_in']
    
    heatmap_data = df.pivot_table(
        index='date', 
        columns='time_float', 
        values='total_traffic',
        aggfunc='sum'
    ).fillna(0)

    # 4. 绘图优化
    # [优化1] 设置绘图上下文为 'talk'，这会自动增大所有元素的字体，适合演示
    sns.set_context("talk", font_scale=1.1)
    
    # [优化] 稍微加大画布尺寸，防止标签拥挤
    plt.figure(figsize=(18, 10))
    sns.set_style("white")
    
    # 绘制热力图
    ax = sns.heatmap(heatmap_data, cmap="coolwarm", robust=True, 
                     cbar_kws={'label': 'Passenger Traffic', 'shrink': 0.8}) # shrink让色条短一点，更精致
    
    # [优化1] 字体加粗加大
    plt.title('Elevator Traffic Intensity (24H Heatmap)', fontsize=24, fontweight='bold', pad=20)
    plt.xlabel('Time of Day (Hour)', fontsize=18, labelpad=10)
    plt.ylabel('Date', fontsize=18, labelpad=10)
    
    # X 轴刻度 (只显示整点)
    xticks = [i for i in range(0, 24)]
    xticks_pos = [heatmap_data.columns.get_loc(x) for x in xticks if x in heatmap_data.columns]
    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticks, rotation=0, fontsize=14)

    # [优化2] Y 轴刻度 (加上星期几)
    all_dates = heatmap_data.index
    # 依然只显示周一和周五，避免太乱
    monday_friday_dates = [date for date in all_dates if date.weekday() in [0, 4]]
    yticks_pos = [heatmap_data.index.get_loc(date) for date in monday_friday_dates]
    
    # 核心修改：格式化日期字符串，加入 (%a) 显示星期缩写
    yticks_labels = [f"{date.strftime('%Y-%m-%d')} ({date.strftime('%a')})" for date in monday_friday_dates]
    
    ax.set_yticks(yticks_pos)
    ax.set_yticklabels(yticks_labels, rotation=0, fontsize=14, va='center')

    # 5. 保存
    IMG_PATH.mkdir(exist_ok=True)
    save_path = IMG_PATH / 'traffic_heatmap_optimized_v2.png'
    
    # [优化3] 使用 bbox_inches='tight' 确保长标签不被裁切
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"🖼️ Optimized heatmap saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_traffic_heatmap()