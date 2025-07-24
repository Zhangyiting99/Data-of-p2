# ======================== 导入库 ========================
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from keras_tuner import RandomSearch
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

# ======================== 数据加载与预处理 ========================
# 加载数据集
data = pd.read_csv('p2data250312.csv')  
df = pd.DataFrame(data)

target = 'target'  # 目标变量
features = df.columns.drop(target)  # 特征变量

X = df[features].values
y = df[target].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.fit_transform(X)

# ======================== 模型构建函数 ========================
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(X_train_scaled.shape[1],)))

    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(Dense(
            units=hp.Int(f'units_{i}', min_value=8, max_value=64, step=8),
            activation='relu'
        ))

    model.add(Dense(1, activation='linear'))

    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    return model

# ======================== Keras Tuner 超参数调优 ========================
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=100,
    executions_per_trial=2,
    directory='tuner_results',
    project_name='nn_tuning',
    overwrite=True
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_val_loss = float('inf')
best_hyperparams = None

for train_idx, val_idx in kf.split(X_train_scaled):
    X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    # 进行超参数搜索
    tuner.search(X_train_fold, y_train_fold, epochs=50, validation_data=(X_val_fold, y_val_fold))

    # 获取最佳超参数并训练
    current_best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(current_best_hp)
    history = model.fit(X_train_fold, y_train_fold, epochs=50, validation_data=(X_val_fold, y_val_fold), verbose=0)

    val_loss = min(history.history['val_loss'])
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_hyperparams = current_best_hp

print("最佳超参数:", best_hyperparams.values)

# ======================== 使用最佳超参数训练最终模型 ========================
final_model = tuner.hypermodel.build(best_hyperparams)
history = final_model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_data=(X_test_scaled, y_test))

# ======================== 预测与评估 ========================
y_train_pred = final_model.predict(X_train_scaled)
y_test_pred = final_model.predict(X_test_scaled)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print(f"训练集 R2 分数: {r2_train:.4f}")
print(f"测试集 R2 分数: {r2_test:.4f}")
print(f"训练集 MSE: {mse_train:.4f}")
print(f"测试集 MSE: {mse_test:.4f}")

# ======================== 可视化真实值 vs 预测值 ========================
plt.figure(figsize=(8, 6), dpi=600)
plt.scatter(y_train, y_train_pred, color='blue', label='训练数据')
plt.scatter(y_test, y_test_pred, color='green', label='测试数据')
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label='理想拟合 (y=x)')
plt.xlabel('真实值', fontsize=16)
plt.ylabel('预测值', fontsize=16)
plt.title('真实值与预测值关系图', fontsize=18)
plt.legend(fontsize=14)
plt.tick_params(axis='both', labelsize=14)
plt.grid(True)
plt.show()

# ======================== 可视化损失曲线 ========================
plt.figure(figsize=(8, 6), dpi=600)
plt.plot(history.history['loss'], label='训练损失', color='blue')
plt.plot(history.history['val_loss'], label='验证损失', color='red')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('训练与验证损失曲线', fontsize=18)
plt.legend(fontsize=14)
plt.tick_params(axis='both', labelsize=14)
plt.grid(True)
plt.show()

# ======================== 保存结果 ========================
# 保存训练集真实值和预测值
train_results = pd.DataFrame({
    'True_Train': y_train,
    'Pred_Train': y_train_pred.ravel()
})
train_results.to_csv('train_results.csv', index=False)

# 保存测试集真实值和预测值
test_results = pd.DataFrame({
    'True_Test': y_test,
    'Pred_Test': y_test_pred.ravel()
})
test_results.to_csv('test_results.csv', index=False)

# 保存最佳超参数
pd.DataFrame([best_hyperparams.values]).to_csv('best_hyperparameters.csv', index=False)

# 9. 保存模型
final_model.save('final_bpnn_model.h5')

# ======================== 绘制结果图 ========================
# 读取训练和测试结果
train_results = pd.read_csv('train_results.csv')
test_results = pd.read_csv('test_results.csv')

y_train = train_results['True_Train'].values
y_train_pred = train_results['Pred_Train'].values
y_test = test_results['True_Test'].values
y_test_pred = test_results['Pred_Test'].values

# 计算 R² 分数
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# 设置画布和子图
fig = plt.figure(figsize=(12, 12))
main_ax = plt.axes([0.2, 0.2, 0.65, 0.65])
top_ax = plt.axes([0.2, 0.852, 0.65, 0.15])
right_ax = plt.axes([0.852, 0.2, 0.15, 0.65])

# 主散点图
main_ax.scatter(y_train, y_train_pred, c='dodgerblue', label='Train', alpha=0.6, s=160)
main_ax.scatter(y_test, y_test_pred, c='lightcoral', label='Test', alpha=0.6, s=160)

# y = x 线
data_min = min(min(y_train), min(y_test), min(y_train_pred), min(y_test_pred))
data_max = max(max(y_train), max(y_test), max(y_train_pred), max(y_test_pred))
main_ax.plot([data_min, data_max], [data_min, data_max], '--', c='gray', label='y = x')

# R² 显示
main_ax.text(0.05, 0.92, f'Backpropagation Neural Network', transform=main_ax.transAxes,
             fontsize=30, verticalalignment='bottom', color='black')

main_ax.text(0.05, 0.90, f'Train $R^2$ = {r2_train:.3f}', transform=main_ax.transAxes,
             fontsize=30, verticalalignment='top', color='dodgerblue')
main_ax.text(0.05, 0.83, f'Test $R^2$ = {r2_test:.3f}', transform=main_ax.transAxes,
             fontsize=30, verticalalignment='top', color='lightcoral')

# 主图标签
main_ax.set_xlabel('True Values', fontsize=48)
main_ax.set_ylabel('Predicted Values', fontsize=48)
main_ax.tick_params(axis='both', labelsize=36)
main_ax.legend(fontsize=36)
main_ax.tick_params(axis='both', which='both', direction='in', width=2, length=8, color='black')
# 设置 bin
bin_width = (data_max - data_min) / 30
bins = np.arange(data_min, data_max + bin_width, bin_width)

# 计算直方图频数以统一坐标轴
hist_train = np.histogram(y_train, bins=bins)[0]
hist_test = np.histogram(y_test, bins=bins)[0]
hist_train_pred = np.histogram(y_train_pred, bins=bins)[0]
hist_test_pred = np.histogram(y_test_pred, bins=bins)[0]

max_top = max(hist_train.max(), hist_test.max())
max_right = max(hist_train_pred.max(), hist_test_pred.max())
common_max = max(max_top, max_right)

# 顶部直方图 + KDE
top_ax.hist(y_train, bins=bins, color='dodgerblue', alpha=0.5, density=False)
top_ax.hist(y_test, bins=bins, color='lightcoral', alpha=0.5, density=False)
top_ax.set_ylim(0, max_top)

# 右侧直方图 + KDE
right_ax.hist(y_train_pred, bins=bins, orientation='horizontal', color='dodgerblue', alpha=0.5, density=False)
right_ax.hist(y_test_pred, bins=bins, orientation='horizontal', color='lightcoral', alpha=0.5, density=False)
right_ax.set_xlim(0, max_right)

# 移除边际图刻度和边框
for ax in [top_ax, right_ax]:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    for spine in ax.spines.values():
        spine.set_visible(False)

# 主图边框加粗
for spine in main_ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')

plt.show()
