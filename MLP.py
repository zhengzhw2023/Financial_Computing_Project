import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# prepare a helper function for plotting prices to avoid code redundancy
def plot_prices(y_pred, y_test, model_name, title="Closing Price Predictions"):
    plt.figure(figsize=(9,6))
    plt.title(title + f" ({model_name})")
    plt.plot(y_pred, label=model_name)
    plt.plot(y_test, label="Actual")
    plt.ylabel('Price')
    plt.xlabel('Day')
    plt.legend()
    plt.show()

# 指定股票代码和时间范围
ticker = 'AAPL'  # Apple股票的代码
start_date = '2019-01-01'
end_date = '2021-12-31'

# 使用yfinance库获取股票数据
data = yf.download(ticker, start=start_date, end=end_date)

# 将价格数据向上移动一个时间步长，即获取下一个价格
data['Next Price'] = data['Close'].shift(-1)


n_values = [1, 2, 3, 4, 5]  # 多个n值

# 获取多个n-Price
for n in n_values:
    data[f'Day _n-{n} Price'] = data['Close'].shift(n)

print("数据",data)
# data.to_csv('apple_stock_data.csv')
#
# # get a quick overview of the data
# # print(data.info())
# # print(data.describe())
#
# get rid of rows with empty values
data = data.dropna()
#
# drop unnecessary columns (only leave closing prices from the previous 5 days and those from the 'Close' column)
# store the result in a variable called X
X = data[["Close"] + list(filter(lambda x: "Day" in x, data.columns))]

print(X)
# # X.to_csv('apple_stock_data.csv')

# prepare outputs by storing values from the 'Next Price' column in a variable called y (preserve the case)
y = data["Next Price"].to_frame()
# # 使用ravel()函数将y转换为一维数组
# # y = np.ravel(y)

# split the data into a training and test sets (50% training, 50% testing)
# remember to set shuffle to False
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, shuffle=False)

# prepare a dictionary for storing test errors and predictions
model_data = dict()
#
# normalise the data to speed up training (only fit on the training set!)
X_scaler, y_scaler = StandardScaler(), StandardScaler()
X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train)
X_test = X_scaler.transform(X_test)
y_test = y_test.values
#
# #
# # # get a quick overview of the data
# # print(data.info())
# # print(data.describe())
#
#
# # print(data.shape)
#
# now let's try a neural network (multi-layer perceptron)
model = MLPRegressor(activation= 'relu',alpha= 0.005, hidden_layer_sizes= (100,), learning_rate= 'invscaling', max_iter= 400, solver= 'lbfgs',batch_size = 128)


# mlp = MLPRegressor()
# # 定义参数网格
# param_grid = {
#     'hidden_layer_sizes': [(100),(50,100,150)],
#     'activation': ['relu'],
#     'solver': ['lbfgs'],
#     'alpha': [ 0.005],
#     'learning_rate': ['invscaling'],
#     'max_iter': [400],
#     'batch_size' : [128],
# }
#
# # 定义GridSearchCV对象
# grid_search = GridSearchCV(mlp, param_grid, cv=5)
#
# # 进行参数搜索
# grid_search.fit(X, y)
#
# # 输出最佳参数组合和对应的性能指标
# print("Best parameters found: ", grid_search.best_params_)
# print("Best score: ", grid_search.best_score_)
#


# train the model
model.fit(X_train, y_train)

# get predictions
normalised_y_pred = model.predict(X_test)

# scale the outputs back
y_pred = y_scaler.inverse_transform(normalised_y_pred.reshape(-1, 1))
print()

# calculate error
model_name = "MLP"
model_data[model_name] = {
    "error": mean_absolute_error(y_test, y_pred),
    "predictions": y_pred
}
print("Error of " + model_name + " regression model:", model_data[model_name]["error"])

# plot the results
plot_prices(y_pred, y_test, model_name)
# #
# #
# # # # 提取股票的收盘价格
# # closing_prices = data['Close']
#
# # 绘制折线图
# plt.plot(closing_prices)
# plt.title('Stock Closing Price')
# plt.xlabel('Date')
# plt.ylabel('Price')
#
# # 显示图表
# plt.show()

