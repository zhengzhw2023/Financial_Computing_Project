import os
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tensorflow import keras
import tensorflow as tf


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



# define helper functions for LSTM training

# this function splits the data into sequences using the sliding window approach
def split_into_windows(data, window_size, output_col, input_cols):
    X, y = [], []
    for i in range(0, data.shape[0] - window_size):
        X.append(data.loc[i:i + window_size - 1, input_cols])
        y.append([data.loc[i + window_size, output_col]])
    return np.array(X), np.array(y)


# this function constructs a Keras LSTM model and returns it
def build_lstm_model(window_size, num_features, num_units):
    if len(num_units) == 0:
        raise Exception("Number of units cannot be empty")
    model = keras.models.Sequential()

    # at least 1 layer is required
    model.add(keras.layers.LSTM(units=num_units[0], input_shape=(window_size, num_features), activation="relu",
                                return_sequences=(True if len(num_units) > 1 else False)))

    # add the specified number of LSTM layers
    for i in range(1, len(num_units)):
        model.add(keras.layers.LSTM(units=num_units[i], activation="relu",
                                    return_sequences=(True if i != len(num_units) - 1 else False)))

        # add dropout for regularisation
        model.add(keras.layers.Dropout(0.2))

    # this is a fully-connected layer, where every node from the previous layer is connected to every node in the next layer
    model.add(keras.layers.Dense(units=1))  # 1 output value

    # compile the model; since this is a regression model, we will use mean_squared_error as our loss function
    model.compile(loss="mean_squared_error", optimizer="adam",
                  metrics=["mae", "mse", tf.keras.metrics.MeanAbsolutePercentageError()])

    return model


def plot_training_history(history):
    history = history.history
    plt.plot(history["val_loss"], label="validation loss")
    plt.plot(history["loss"], label="training loss")
    plt.legend()
    plt.figure()

    if "lr" in history:
        plt.plot(history["lr"], label="learning rate")
        plt.legend()
        plt.figure()



if __name__ == '__main__':
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

    print("数据", data)
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

    # set model parameters
    output_col = "Close"  # because we're trying to predict closing prices in the future
    features = ["Close"]  # + list(filter(lambda x: "Day" in x, data.columns))
    window_size = 5  # play around with window size to see its effect on model performance
    train_ratio = .5
    training_set_size = int(train_ratio * data.shape[0])
    # try adjusting the number of layers and hidden units in each layer;
    # len(hidden_layers) = number of layers; hidden_layers[i] = number of hidden units in layer i
    # we shall try 50 units in 2 hidden layers followed by a dense layer
    hidden_layers = [50, 50]
    epochs = 10
    batch_size = 64  # preferably an exponent of 2, e.g. 16, 32, 64, 128

    # prepare the training dataset
    dataset_cols = list(set(features + [output_col]))
    training_dataset = data[:training_set_size][dataset_cols]

    # rearrange input data into windows
    X_train, y_train = split_into_windows(training_dataset, window_size, output_col, features)

    # print training set shapes
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    # prepare the test set
    testing_dataset = data[training_set_size - window_size:][list(set(features + [output_col]))].reset_index(drop=True)

    # split into windows
    X_test, y_test = split_into_windows(testing_dataset, window_size, output_col, features)

    # print test set shapes
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    model_name = f"{output_col}_" + "_".join([str(i) for i in hidden_layers]) + f"_{window_size}_{epochs}"

    if os.path.isdir(model_name):
        # if the model is already in local disk, just load it
        model = keras.models.load_model(model_name)
    else:
        # construct the model
        model = build_lstm_model(window_size, len(features), hidden_layers)

        # start training
        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), shuffle=False,
                            batch_size=batch_size)

        # save the model so that we do not have to train it every time
        model.save(model_name)

        # plot the learning rate
        plot_training_history(history)

    # get y_test from the testing dataset
    y_test = testing_dataset[[output_col]][window_size:].values

    # run the model
    y_pred = model.predict(X_test)

    assert y_test.shape == y_pred.shape

    model_name = "LSTM"
    model_data[model_name] = {
        "error": mean_absolute_error(y_test, y_pred),
        "predictions": y_pred
    }
    print("Error of " + model_name + " regression model:", model_data[model_name]["error"])

    # calculate total error
    print("Error of " + model_name + " regression model:", )

    plot_prices(y_pred, y_test, model_name)