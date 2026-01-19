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
    # 1. 读取清洗好的 5分钟流量表
    df = pd.read_csv(PROCESSED_PATH / 'traffic_5min.csv')
    df['Time'] = pd.to_datetime(df['Time'])
    
    # 2. 特征提取：我们需要 "Hour" (几点) 和 "Date" (哪天)
    df['hour'] = df['Time'].dt.hour
    df['date'] = df['Time'].dt.date
    df['minute'] = df['Time'].dt.minute
    
    # 为了热力图好看，我们把时间搞成浮点数，比如 9.5 代表 9:30
    df['time_float'] = df['hour'] + df['minute']/60
    
    # 3. 构造透视表 (Pivot Table)
    # 行(Index)是具体的“日期”，列(Column)是“时刻”
    # 值(Values)是“总人流量 (hall_call + people_in)”
    df['total_traffic'] = df['hall_call_count'] + df['people_in']
    
    heatmap_data = df.pivot_table(
        index='date', 
        columns='time_float', 
        values='total_traffic',
        aggfunc='sum'
    ).fillna(0)

    # 4. 绘图 (Science 风格)
    plt.figure(figsize=(15, 8))
    sns.set_theme(style="white")
    
    # 绘制热力图 (使用 'coolwarm' 颜色方案，对比度更高)
    ax = sns.heatmap(heatmap_data, cmap="coolwarm", robust=True, cbar_kws={'label': 'Passenger Traffic'})
    
    plt.title('Elevator Traffic Intensity (24H Heatmap)', fontsize=16, fontweight='bold')
    plt.xlabel('Time of Day (Hour)', fontsize=12)
    plt.ylabel('Date', fontsize=12)
    
    # 优化 X 轴刻度 (只显示整点)
    xticks = [i for i in range(0, 24)]
    # 计算每个整点在 pivot table 列中的位置
    xticks_pos = [heatmap_data.columns.get_loc(x) for x in xticks if x in heatmap_data.columns]
    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticks, rotation=0) # 旋转角度为0，横着写

    # 优化 Y 轴刻度 (只显示每周一和周五的日期)
    # 获取所有日期
    all_dates = heatmap_data.index
    # 筛选出周一和周五的日期
    monday_friday_dates = [date for date in all_dates if date.weekday() in [0, 4]]
    # 计算这些日期在 pivot table 行中的位置
    yticks_pos = [heatmap_data.index.get_loc(date) for date in monday_friday_dates]
    ax.set_yticks(yticks_pos)
    ax.set_yticklabels(monday_friday_dates, rotation=0) # 旋转角度为0，横着写

    # 5. 保存
    IMG_PATH.mkdir(exist_ok=True)
    save_path = IMG_PATH / 'traffic_heatmap_optimized.png' # 保存为新文件名
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"🖼️ Optimized heatmap saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_traffic_heatmap()