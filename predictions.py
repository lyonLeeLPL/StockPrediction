# Standard packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
import os
os.environ['KMP_WARNINGS'] = 'off'

from sklearn.preprocessing import MinMaxScaler


class Predict(object):
    def __init__(self):
        pass

    def train(self, train_data, model, show_progress = True):
        '''
            Input: train_data - list of input values (numpy array) and target values
                                (numpy array) of training data
                   model - model to be trained
                   show_progress - if the training process is showed (boolean)

        '''


        self.x_train = train_data[0]
        self.y_train = train_data[1]
        history = model.fit(self.x_train, self.y_train, batch_size = 1, epochs = 1)
        self.model = model

       # if show_progress:
       #     plt.plot(history.history['loss'])
       #     plt.title('RMS loss')
       #     plt.ylabel('$\mathcal{L}$')
       #     plt.xlabel('epoch')
       #     plt.grid('on')
       #     plt.show()


    def predict(self, test_data, scaler, data_scaled = True, show_predictions = True):
        '''
            Input: test_data - list of input values (numpy array) and target values
                               (numpy array) of validation data
                   scaler - scaler object to inversely scale predictions
                   data_scaled - if scaler were used in the preprocessing (boolean)

            Output: predictions - numpy array of the predicted values
        '''

        try:
            model = self.model
        except Exception as e:
            print('Unable to load model, train the model before predicting!')
            return None

        self.x_test = test_data[0]
        self.y_test = test_data[1]
        predictions = model.predict(self.x_test)
        if data_scaled:
            predictions = scaler.inverse_transform(predictions)

        #rmse = np.sqrt(np.mean(((predictions - self.y_test) ** 2)))

        return predictions

def plot_predictions(df, train_data_size, predictions):
    '''
        Input: df - dataframe of stock values
               train_data_size - length of the training data, number of elements (int)
               predictions - numpy array of the prdicted values
    '''
    colors = ['#579BF5', '#C694F6', '#F168F1']
    fig = go.Figure()
    train = df[:train_data_size]
    valid = df[train_data_size:]
    valid['Predictions'] = predictions
    x_train = [str(train.index[i]).split()[0] for i in range(len(train))]
    x_val = [str(valid.index[i]).split()[0] for i in range(len(valid))]

    fig.add_trace(
        go.Scatter(x=x_train, y=train['Adj Close'], mode='lines', line_color=colors[0], line_width=3,
                   name='Training data'))

    fig.add_trace(
        go.Scatter(x=x_val, y=valid['Adj Close'], mode='lines', line_color=colors[1], line_width=3,
                   name='Validation data'))

    fig.add_trace(
        go.Scatter(x=x_val, y=valid['Predictions'], mode='lines', line_color=colors[2], line_width=3,
                   name='Predictions'))

    fig.update_layout(showlegend=True)
    fig.update_layout(title=dict(text=f'Predictions of stock "{train["Company stock name"][0]}" from {x_val[0]} to {x_val[len(valid) - 1]}',
                                 xanchor='auto'),
                      xaxis=go.layout.XAxis(
                          title=go.layout.xaxis.Title(
                              text="Date")),
                      yaxis=go.layout.YAxis(
                          title=go.layout.yaxis.Title(
                              text="Adjusted closing price USD ($)"))
                      )
    fig.show()











