# ======================================================
# 外卖分析AI预测 - MAC 环境运行版本（无图形界面显示）
# 需要提前安装依赖库：pip install pandas numpy matplotlib scikit-learn openpyxl
# ======================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无图形界面模式
import matplotlib.pyplot as plt
plt.ioff()

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

# ==============================================
# 1. 加载数据 (GBK编码)
# ==============================================
print("正在加载数据...")

cpc = pd.read_csv("cpc.csv", encoding="gbk")
orders = pd.read_csv("orders.csv", encoding="gbk")
shop = pd.read_csv("shop.csv", encoding="gbk")

print(f"cpc.csv: {len(cpc)} 行")
print(f"orders.csv: {len(orders)} 行")
print(f"shop.csv: {len(shop)} 行")

# ==============================================
# 2. 合并数据（使用中文列名）
# ==============================================
df = pd.merge(cpc, orders, on=["门店ID"])
df = pd.merge(df, shop, on=["门店ID", "日期"])
print(f"合并后数据共 {len(df)} 行")

# ==============================================
# 3. 清洗数据
# ==============================================
df = df[(df["cpc总费用"] > 0) & (df["门店实收"] > 0)].dropna()
print(f"清洗后数据共 {len(df)} 行")

# 保存合并数据到Excel
df.to_excel("外卖分析预测合并数据.xlsx", index=False)
print("已生成 Excel 文件")

# ==============================================
# 图片 1: 广告费用-销售额关系（散点图）
# ==============================================
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(df["cpc总费用"], df["门店实收"], alpha=0.5)
ax.set_title("广告费用与销售额关系")
ax.set_xlabel("广告费用")
ax.set_ylabel("门店实收")
ax.grid(True)
plt.tight_layout()
plt.savefig("img1.png", dpi=150)
plt.close()
print("已生成图片1: 广告费用-销售额散点图 (img1.png)")

# ==============================================
# 图片 2: 城市订单数量
# ==============================================
city_order = df.groupby("城市_x")["有效订单"].mean()
fig, ax = plt.subplots(figsize=(10,5))
city_order.plot(kind="bar", ax=ax)
ax.set_title("城市平均有效订单数")
plt.tight_layout()
plt.savefig("img2.png", dpi=150)
plt.close()
print("已生成图片2: 城市订单数量柱状图 (img2.png)")

# ==============================================
# 图片 3: 品类价格分布（品牌名称）
# ==============================================
cat_price = df.groupby("品牌名称_x")["单均gmv"].mean()
fig, ax = plt.subplots(figsize=(10,5))
cat_price.plot(kind="bar", ax=ax)
ax.set_title("品牌单均GMV")
plt.tight_layout()
plt.savefig("img3.png", dpi=150)
plt.close()
print("已生成图片3: 品类价格分布柱状图 (img3.png)")

# ==============================================
# AI预测
# ==============================================
le_city = LabelEncoder()
le_cat = LabelEncoder()
df["城市编码"] = le_city.fit_transform(df["城市_x"])
df["品牌编码"] = le_cat.fit_transform(df["品牌名称_x"])

X = df[["cpc总费用", "城市编码", "品牌编码"]]
y = df["门店实收"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# ==============================================
# 图片 4: 预测结果对比
# ==============================================
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(y_test, y_pred, alpha=0.6)
ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
ax.set_title("真实值与预测值对比")
ax.set_xlabel("真实销售额")
ax.set_ylabel("预测销售额")
plt.tight_layout()
plt.savefig("img4.png", dpi=150)
plt.close()
print("已生成图片4: 真实值与预测值对比图 (img4.png)")

# ==============================================
# 输出预测结果
# ==============================================
print("\n===== 预测结果 =====")
print(f"R² 回归系数为 {r2:.2f}")

print("\n===== 预测 =====")
for cost in [200, 500, 800, 1000]:
    pred = model.predict([[cost, 0, 0]])[0]
    print(f"投入{cost}元广告费，预计销售额{int(pred)}元")

print("\n分析完成！已成功生成4张图片和1份Excel文件！")
