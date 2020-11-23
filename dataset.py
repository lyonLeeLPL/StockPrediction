
# Standard packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
import random


# Collect data from yahoo finance
from pandas_datareader.data import DataReader

# For time stamps
from datetime import datetime


from sklearn.preprocessing import MinMaxScaler


class Dataset(object):
    def __init__(self, df, feature='Adj Close'):
        super(Dataset, self).__init__()
        self.df = df
        self.feature = feature


    def get_dataset(self, scale=True):
        data = self.df.filter([str(self.feature)])
        self.data_values = data.values
        if scale:
            scaler = MinMaxScaler(feature_range=(0, 1))
            self.dataset = scaler.fit_transform(self.data_values)

        else:
            self.dataset = self.data_values

        return self.dataset

    def get_size(self):
        return len(self.dataset)


    def split(self, train_split_ratio = 0.8, time_period = 30):
        train_data_size = int(np.ceil(self.get_size() * train_split_ratio))
        self.train_data = self.dataset[0:int(train_data_size), :]
        x_train, y_train = [], []
        for i in range(time_period, len(self.train_data)):
            x_train.append(self.train_data[i-time_period:i, 0])
            y_train.append(self.train_data[i, 0])

        self.y_train = np.array(y_train)
        self.x_train = np.reshape(np.array(x_train), np.array(x_train).shape[0], np.array(x_train).shape[1], 1)
        print(f'Shape of train data: (x, y) = ({np.shape(self.x_train)}, {np.shape(self.y_train)})')

        self.test_data = self.dataset[train_data_size - time_period:, :]
        x_test = []
        self.y_test = self.dataset[train_data_size:, :]
        for i in range(time_period, len(self.test_data)):
            x_test.append(self.test_data[i - time_period:i, 0])

        self.x_test = np.reshape(np.array(x_test), (np.array(x_test).shape[0], np.array(x_test).shape[1], 1))
        return [self.x_train, self.y_train], [self.x_test, self.y_test]




