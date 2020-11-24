
# Standard packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

from sklearn.preprocessing import MinMaxScaler


class Dataset(object):
    def __init__(self, df, feature='Adj Close'):
        super(Dataset, self).__init__()
        self.df = df
        self.feature = feature


    def get_dataset(self, scale=True):
        '''
            Input: scale - if to scale the input data
        '''
        data = self.df.filter([str(self.feature)])
        self.data_values = data.values
        if scale:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.dataset = self.scaler.fit_transform(self.data_values)

        else:
            self.dataset = self.data_values


    def get_size(self):
        '''
            Output: returns the length of the dataset
        '''
        return len(self.dataset)


    def split(self, train_split_ratio = 0.8, time_period = 30):
        '''
            Input: train_split_ratio - percentage of dataset to be used for
                                       the training data (float)
                   time_period - time span in days to be predicted (in)

            Output: lists of the training and validation data (input values and target values)
                    size of the training data
        '''
        train_data_size = int(np.ceil(self.get_size() * train_split_ratio))
        self.train_data = self.dataset[0:int(train_data_size), :]
        x_train, y_train = [], []
        for i in range(time_period, len(self.train_data)):
            x_train.append(self.train_data[i-time_period:i, 0])
            y_train.append(self.train_data[i, 0])

        self.y_train = np.array(y_train)
        self.x_train = np.reshape(np.array(x_train), (np.array(x_train).shape[0], np.array(x_train).shape[1], 1))
        print(f'Shape of train data: (x, y) = ({np.shape(self.x_train)}, {np.shape(self.y_train)})')

        self.test_data = self.dataset[train_data_size - time_period:, :]
        x_test = []
        self.y_test = self.dataset[train_data_size:, :]
        for i in range(time_period, len(self.test_data)):
            x_test.append(self.test_data[i - time_period:i, 0])

        self.x_test = np.reshape(np.array(x_test), (np.array(x_test).shape[0], np.array(x_test).shape[1], 1))
        print(f'Shape of test data: (x, y) = ({np.shape(self.x_test)}, {np.shape(self.y_test)})')
        return [self.x_train, self.y_train], [self.x_test, self.y_test], train_data_size




