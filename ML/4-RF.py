import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 读取数据
data = pd.read_csv('p2data250312.csv')
target = 'target'
features = data.columns.drop(target)

X = data[features].values
y = data[target].values

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 保存训练和测试数据
joblib.dump(X_train_scaled, 'X_train_scaled.pkl')
joblib.dump(y_train, 'y_train.pkl')
joblib.dump(X_test_scaled, 'X_test_scaled.pkl')
joblib.dump(y_test, 'y_test.pkl')

# 随机森林超参数空间
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

# 建模型
rf = RandomForestRegressor(random_state=42)

# 随机搜索调参，5折交叉验证，评价指标R2
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,
    scoring='r2',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_scaled, y_train)

best_params = random_search.best_params_
print("最佳参数:", best_params)
print("最佳交叉验证R2:", random_search.best_score_)

# 保存最佳超参数
with open('best_params_rf.json', 'w') as f:
    json.dump(best_params, f, indent=4)

# 用最佳参数训练最终模型
best_rf = random_search.best_estimator_
best_rf.fit(X_train_scaled, y_train)

# 预测
y_train_pred = best_rf.predict(X_train_scaled)
y_test_pred = best_rf.predict(X_test_scaled)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f"训练集R²: {r2_train:.4f}")
print(f"测试集R²: {r2_test:.4f}")

# 保存模型和Scaler
joblib.dump(best_rf, 'best_rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# 画图函数
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

# 调用画图
plot_results(y_train, y_train_pred, y_test, y_test_pred, r2_train, r2_test, "Random Forest")
