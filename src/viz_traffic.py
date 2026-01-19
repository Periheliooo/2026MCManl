import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# 设置路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
PROCESSED_PATH = project_root / "data" / "processed"
IMG_PATH = project_root / "images"

def plot_traffic_heatmap():
    # 1. 读取数据
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
    sns.set_context("talk", font_scale=1.1)
    
    plt.figure(figsize=(18, 10))
    sns.set_style("white")
    
    # 绘制热力图
    ax = sns.heatmap(heatmap_data, cmap="coolwarm", robust=True, 
                     cbar_kws={'label': 'Passenger Traffic', 'shrink': 0.8})
    
    # 标题优化
    plt.title('Elevator Traffic Intensity (24H Heatmap)', fontsize=24, fontweight='bold', pad=20)
    plt.xlabel('Time of Day (Hour)', fontsize=18, labelpad=10)
    plt.ylabel('Date', fontsize=18, labelpad=10)
    
    # --- X 轴优化 (保持不变) ---
    xticks = [i for i in range(0, 24)]
    xticks_pos = [heatmap_data.columns.get_loc(x) for x in xticks if x in heatmap_data.columns]
    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticks, rotation=0, fontsize=14)

    # --- Y 轴深度优化 (核心修改) ---
    all_dates = heatmap_data.index
    monday_friday_dates = [date for date in all_dates if date.weekday() in [0, 4]]
    
    # 修正1：获取索引位置后 +0.5，让标签对齐到格子的“垂直中心”，而不是格子的“顶部边缘”
    yticks_pos = [heatmap_data.index.get_loc(date) + 0.5 for date in monday_friday_dates]
    
    # 修正2：使用 formatted string 保证日期格式一致
    yticks_labels = [f"{date.strftime('%Y-%m-%d')} ({date.strftime('%a')})" for date in monday_friday_dates]
    
    ax.set_yticks(yticks_pos)
    
    # 修正3：设置 fontfamily='monospace' (等宽字体)
    # 这能保证 '2025-11-03' 和 '2025-11-14' 在视觉上严格对齐，不会出现参差不齐
    ax.set_yticklabels(
        yticks_labels, 
        rotation=0, 
        fontsize=14, 
        va='center',      # 垂直居中
        fontfamily='monospace' # 等宽字体，解决左侧参差不齐的问题
    )

    # 5. 保存
    IMG_PATH.mkdir(exist_ok=True)
    save_path = IMG_PATH / 'traffic_heatmap_final.png'
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"🖼️ Optimized heatmap saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_traffic_heatmap()