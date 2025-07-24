# ======================== 导入库 ========================
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl

# ======================== 全局设置 ========================
mpl.rcParams.update({
    'font.family': 'Arial',  # 设置字体为Arial
    'axes.unicode_minus': False,  # 解决负号显示问题
    'font.size': 30,  # 设置字体大小
    'axes.titlesize': 38,  # 设置标题字体大小
    'axes.labelsize': 34,  # 设置坐标轴标签字体大小
    'xtick.labelsize': 30,  # 设置x轴刻度标签字体大小
    'ytick.labelsize': 30,  # 设置y轴刻度标签字体大小
    'legend.fontsize': 30,  # 设置图例字体大小
})

# ======================== 加载模型和数据 ========================
final_model = load_model('final_bpnn_model.h5')  # 加载训练好的BPNN模型
final_model.summary()  # 输出模型结构

# 加载数据集
data = pd.read_csv('p2data250312.csv')
df = pd.DataFrame(data)

# 分离特征和目标变量
target = 'target'
features = df.columns.drop(target)  # 获取特征列
X = df[features].values  # 获取特征数据

# 标准化特征数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 对特征数据进行标准化

# ======================== 计算SHAP值 ========================
masker = shap.maskers.Partition(X_scaled, clustering="correlation", max_samples=200)  # 设置SHAP解释器的掩码
explainer = shap.Explainer(final_model.predict, masker)  # 使用模型和掩码创建SHAP解释器
shap_values = explainer(X_scaled)  # 计算SHAP值

# 提取SHAP值数组
shap_values_array = shap_values.values
if len(shap_values_array.shape) == 3:
    shap_values_array = shap_values_array[0]  # 如果SHAP值的形状是三维，取第一个维度
if len(shap_values_array.shape) != 2:
    raise ValueError("SHAP值必须是二维数组。")  # 如果SHAP值不是二维数组，抛出错误

# ======================== 可视化SHAP值 ========================
# 1. 汇总柱状图
shap.summary_plot(
    shap_values_array,
    X_scaled,
    feature_names=features,
    plot_type="bar",
    plot_size=(12, 8)
)
plt.tight_layout()  # 调整布局

# 2. 蜂巢图
shap.plots.beeswarm(shap_values)  # 绘制蜂巢图
plt.tight_layout()

# 3. 第一个特征的依赖图
shap.dependence_plot(features[0], shap_values_array, X_scaled, feature_names=features)
plt.tight_layout()

# ======================== 特征重要性分析 ========================
shap_values_mean = np.abs(shap_values_array).mean(axis=0)  # 计算每个特征的平均SHAP值（绝对值）
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': shap_values_mean
}).sort_values(by='Importance', ascending=False)  # 按重要性排序特征

# ======================== 环形图 ========================
plt.figure(figsize=(12, 12))

total_importance = feature_importance['Importance'].sum()  # 计算特征重要性的总和
proportions = feature_importance['Importance'] / total_importance  # 计算每个特征的重要性占比

# 统一的颜色调色板
colors = plt.cm.Set3(np.linspace(0, 1, len(feature_importance)))

wedges, texts, autotexts = plt.pie(
    proportions,
    labels=feature_importance['Feature'],
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops={'width': 0.5, 'edgecolor': 'white', 'linewidth': 2},
    pctdistance=0.8,
    textprops={'fontsize': 30}
)

plt.setp(autotexts, color='white', weight='bold', fontsize=34)  # 设置百分比文本的样式
plt.setp(texts, fontsize=34)  # 设置标签的字体大小

centre_circle = plt.Circle((0, 0), 0.35, fc='white')  # 添加中心白色圆形，形成环形图
plt.gca().add_artist(centre_circle)

plt.title('Feature Importance Donut Chart', fontsize=38, pad=20)
plt.tight_layout()  # 调整布局
plt.show()


# ======================== 输出结果 ========================
print("\nFeature Importance Analysis:")
print(feature_importance)  # 输出特征重要性分析结果

print("\nSensitivity Analysis:")
sensitivity_analysis = feature_importance.copy()
sensitivity_analysis.columns = ['Feature', 'Sensitivity']
print(sensitivity_analysis.sort_values(by='Sensitivity', ascending=False))  # 输出敏感度分析结果

# ======================== 合并蜂巢图和柱状图 ========================
# 创建一个主图，用于合并蜂巢图和柱状图
fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)  # 增大图形尺寸

# 1. 绘制蜂巢图
shap.summary_plot(shap_values, X_scaled, feature_names=features, plot_type='dot', show=False, color_bar=True)
plt.gca().set_position([0.01, 0.15, 0.85, 0.85])  # 调整位置，留出空间给色条

# 获取共享的y轴
ax1 = plt.gca()

# 2. 创建共享y轴的第二个图，绘制柱状图
ax2 = ax1.twiny()
shap.summary_plot(shap_values, X_scaled, plot_type="bar", show=False)
ax1.tick_params(axis='both', which='both', direction='in', width=1, length=4, color='black')
ax2.tick_params(axis='both', which='both', direction='in', width=1, length=4, color='black')

# 调整柱状图的位置，使其与蜂巢图对齐
plt.gca().set_position([0.01, 0.15, 0.85, 0.85])

# 设置柱状图的透明度
bars = ax2.patches  # 获取所有柱状图对象
for bar in bars:
    bar.set_alpha(0.2)  # 设置透明度

# 设置x轴标签
ax1.set_xlabel('Shapley Value Contribution', fontsize=20, fontname='Arial')
ax2.set_xlabel('Mean Shapley Value', fontsize=20, fontname='Arial')

# 将顶部的x轴移动到顶部，避免与底部x轴重叠
ax2.xaxis.set_label_position('top')  # 将标签移到顶部
ax2.xaxis.tick_top()  # 将刻度线也移到顶部

# 设置y轴标签
ax1.set_ylabel('Features', fontsize=16, fontname='Arial')
ax1.set_yticklabels(features, fontsize=12, rotation=0, fontname='Arial')  # 显示特征名称，调整字体大小和旋转

# 设置边框线
for spine in ['top', 'right', 'left', 'bottom']:
    ax1.spines[spine].set_visible(True)
    ax1.spines[spine].set_color('black')
    ax1.spines[spine].set_linewidth(1.5)

for spine in ['top']:
    ax2.spines[spine].set_visible(True)
    ax2.spines[spine].set_color('black')
    ax2.spines[spine].set_linewidth(1.5)

# 调整图形的边距，以确保完整显示
plt.subplots_adjust(left=0.2, right=0.9, top=0.8, bottom=0.2)

# 保存合并后的图形为高质量PDF
plt.savefig("SHAP_combined_with_top_line_high_quality_final.pdf", format='pdf', bbox_inches='tight')

# 显示合并后的图形
plt.show()
