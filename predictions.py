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


class Predict:

    def train(self, train_data, model, show_progress = True):
        self.x_train = train_data[0]
        self.y_train = train_data[1]
        history = model.fit(self.x_train, self.y_train, batch_size = 1, epochs = 1)
        self.model = model

        if show_progress:
            plt.plot(history.history['loss'])
            plt.title('RMS loss')
            plt.ylabel('$\mathcal{L}$')
            plt.xlabel('epoch')
            plt.grid('on')
            plt.show()


    def predict(self, test_data, data_scaled = True, show_predictions = True):

        try:
            model = self.model
            print('Train model before predictions....')

        except Exception as e:
            print('Unable to load model, train the model before predicting!')

            return None

        self.x_test = test_data[0]
        self.y_test = test_data[1]
        predictions = model.predict(self.x_test)
        if data_scaled:
            scaler = MinMaxScaler(feature_range=(0, 1))
            predictions = scaler.inverse_transform(predictions)

        rmse = np.sqrt(np.mean(((predictions - self.y_test) ** 2)))







