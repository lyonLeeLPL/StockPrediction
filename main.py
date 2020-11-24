# Standard packages
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# Scripts
from preprocess import config, get_timestamps, collect_data, plot_closing, plot_gain, compare_stocks
from models import keras_lstm
from dataset import Dataset
from predictions import Predict, plot_predictions

def visualization():
    for idx, stock in enumerate(config.stock_names):
        timestamps = get_timestamps(config.yrs, config.mths, config.dys)
        df = collect_data(timestamps, stock, config.moving_averages, True)
        fig1 = plot_closing(df, moving_averages=True, intervals=None)
        fig1.show()
        fig2 = plot_gain(df)
        fig2.show()


def make_predictions(features):
    timestamps = get_timestamps(config.yrs, config.mths, config.dys)
    if len(config.stock_names) == 1:
        for feature in features:
            df = collect_data(timestamps, config.stock_names[0], moving_averages = config.moving_averages, include_gain=True)
            dataset = Dataset(df, feature = feature)
            dataset.get_dataset(scale=True)
            train_data, test_data, train_data_len = dataset.split(train_split_ratio = 0.8, time_period = 30)
            x_train, y_train = train_data
            x_test, y_test = test_data
            model = keras_lstm(x_train)
            pred = Predict()
            pred.train([x_train, y_train], model, show_progress=True)
            scaler = dataset.scaler
            predictions= pred.predict([x_test, y_test], scaler, data_scaled=True, show_predictions=True)
            plot_predictions(df, train_data_len, predictions)



if __name__ == '__main__':
    features = ['Adj Close']
    visualization()
    make_predictions(features)













