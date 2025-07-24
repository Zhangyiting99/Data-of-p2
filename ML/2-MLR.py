import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 绘图函数
def plot_results(y_train, y_train_pred, y_test, y_test_pred, r2_train, r2_test, model_name):
    fig = plt.figure(figsize=(12, 12))
    main_ax = plt.axes([0.2, 0.2, 0.65, 0.65])
    top_ax = plt.axes([0.2, 0.852, 0.65, 0.15])
    right_ax = plt.axes([0.852, 0.2, 0.15, 0.65])
    main_ax.tick_params(axis='both', which='both', direction='in', width=2, length=8, color='black')
    main_ax.scatter(y_train, y_train_pred, c='dodgerblue', label='Train', alpha=0.6, s=160)
    main_ax.scatter(y_test, y_test_pred, c='lightcoral', label='Test', alpha=0.6, s=160)

    data_min = min(min(y_train), min(y_test), min(y_train_pred), min(y_test_pred))
    data_max = max(max(y_train), max(y_test), max(y_train_pred), max(y_test_pred))
    main_ax.plot([data_min, data_max], [data_min, data_max], '--', c='gray', label='y = x')

    main_ax.text(0.05, 0.92, f'{model_name}', transform=main_ax.transAxes,
                 fontsize=30, verticalalignment='bottom', color='black')
    main_ax.text(0.05, 0.90, f'Train $R^2$ = {r2_train:.3f}', transform=main_ax.transAxes,
                 fontsize=30, verticalalignment='top', color='dodgerblue')
    main_ax.text(0.05, 0.83, f'Test $R^2$ = {r2_test:.3f}', transform=main_ax.transAxes,
                 fontsize=30, verticalalignment='top', color='lightcoral')

    main_ax.set_xlabel('True Values', fontsize=48)
    main_ax.set_ylabel('Predicted Values', fontsize=48)
    main_ax.tick_params(axis='both', labelsize=36)
    main_ax.legend(fontsize=36)

    bin_width = (data_max - data_min) / 30
    bins = np.arange(data_min, data_max + bin_width, bin_width)

    hist_train = np.histogram(y_train, bins=bins)[0]
    hist_test = np.histogram(y_test, bins=bins)[0]
    hist_train_pred = np.histogram(y_train_pred, bins=bins)[0]
    hist_test_pred = np.histogram(y_test_pred, bins=bins)[0]

    max_top = max(hist_train.max(), hist_test.max())
    max_right = max(hist_train_pred.max(), hist_test_pred.max())
    common_max = max(max_top, max_right)

    top_ax.hist(y_train, bins=bins, color='dodgerblue', alpha=0.5, density=False)
    top_ax.hist(y_test, bins=bins, color='lightcoral', alpha=0.5, density=False)
    top_ax.set_ylim(0, max_top)

    right_ax.hist(y_train_pred, bins=bins, orientation='horizontal', color='dodgerblue', alpha=0.5, density=False)
    right_ax.hist(y_test_pred, bins=bins, orientation='horizontal', color='lightcoral', alpha=0.5, density=False)
    right_ax.set_xlim(0, max_right)

    for ax in [top_ax, right_ax]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for spine in main_ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    plt.show()


# 1. 加载数据
data = pd.read_csv("p2data250312.csv")  
X = data.drop("target", axis=1).values  
y = data["target"].values

# 2. 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 建立模型
model = LinearRegression()

# 5. 交叉验证评估
cv_results = cross_validate(model, X_train_scaled, y_train, cv=5,
                            scoring=['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error'])

print("===== 交叉验证评估结果（5折）=====")
print("平均 R²:", np.mean(cv_results['test_r2']))
print("平均 RMSE:", -np.mean(cv_results['test_neg_root_mean_squared_error']))
print("平均 MAE:", -np.mean(cv_results['test_neg_mean_absolute_error']))

# 6. 训练并测试
model.fit(X_train_scaled, y_train)
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

print("\n===== 测试集评估结果 =====")
r2_test = r2_score(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
print("测试集 R²:", r2_test)
print("测试集 RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
print("测试集 MAE:", mean_absolute_error(y_test, y_test_pred))

# 7. 绘图展示
plot_results(y_train, y_train_pred, y_test, y_test_pred, r2_train, r2_test, model_name='Multiple Linear Regression')
