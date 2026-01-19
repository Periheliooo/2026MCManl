import pandas as pd
from prophet import Prophet
from pathlib import Path
import matplotlib.pyplot as plt
import joblib  # 用于保存模型

# ============================
# 1. 路径配置
# ============================
current_dir = Path(__file__).resolve().parent
PROCESSED_PATH = current_dir.parent / "data" / "processed"
IMG_PATH = current_dir.parent / "images"
MODEL_PATH = current_dir.parent / "models"  # 新建一个 models 文件夹存模型

# 确保文件夹存在
IMG_PATH.mkdir(exist_ok=True)
MODEL_PATH.mkdir(exist_ok=True)

def train_traffic_model():
    print("🔮 [Prophet] Loading Data & Training...")
    
    # --- A. 数据准备 ---
    traffic_file = PROCESSED_PATH / 'traffic_5min.csv'
    if not traffic_file.exists():
        raise FileNotFoundError("请先运行 src/data_loader.py 生成数据！")
        
    df = pd.read_csv(traffic_file)
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Prophet 需要两列: ds (时间), y (目标值)
    # 我们预测 'total_traffic' (Hall Call + Load In)
    df['y'] = df['hall_call_count'] + df['people_in']
    df['ds'] = df['Time']
    
    # --- B. 模型配置 (关键！) ---
    # daily_seasonality=True: 会自动拟合出“早高峰-午高峰-晚高峰”的波形
    # changepoint_prior_scale: 灵活性参数，默认0.05。调大可以更敏感地捕捉午餐突增
    model = Prophet(
        daily_seasonality=True, 
        weekly_seasonality=True,
        changepoint_prior_scale=0.1 
    )
    
    model.fit(df[['ds', 'y']])
    
    # --- C. 保存模型 (持久化) ---
    # 这样仿真器(Simulator)就可以直接加载它，不用每次都重新训练
    model_file = MODEL_PATH / 'prophet_model.pkl'
    joblib.dump(model, model_file)
    print(f"💾 Model saved to {model_file}")

    # --- D. 验证与绘图 ---
    print("📈 Generating Forecast & Components...")
    
    # 预测未来 24 小时 (288个 5分钟)
    future = model.make_future_dataframe(periods=288, freq='5min')
    forecast = model.predict(future)
    
    # 1. 总体预测图
    fig1 = model.plot(forecast)
    plt.title("Traffic Forecast (Next 24 Hours)")
    plt.xlabel("Date")
    plt.ylabel("Passenger Flow")
    plt.savefig(IMG_PATH / 'pred_overview.png', dpi=300)
    
    # 2. 成分分解图 (这是重点！)
    # 这张图会包含 Trend(趋势), Weekly(周效应), Daily(日效应)
    # 你需要检查 "Daily" 子图，看它是否在 12:00 处有一个高峰
    fig2 = model.plot_components(forecast)
    plt.savefig(IMG_PATH / 'pred_components.png', dpi=300)
    
    print(f"✅ Visualization saved to {IMG_PATH}")
    
    # --- E. 打印午餐时段预测值 (Sanity Check) ---
    # 找一下明天中午 12:00 的预测值
    tomorrow_noon = forecast[forecast['ds'].dt.hour == 12].iloc[0]
    print(f"\n🔍 [Sanity Check] Predicted Traffic for {tomorrow_noon['ds']}:")
    print(f"   Value: {tomorrow_noon['yhat']:.2f} passengers / 5min")
    print("   如果这个值很低，说明模型没学好；如果很高，说明模型捕捉到了午餐高峰。")

if __name__ == "__main__":
    train_traffic_model()