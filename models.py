from keras.models import Sequential
from keras.layers import Dense, LSTM

def keras_lstm(x_train):
    '''
        Input: x_train - numpy array of the training set

        Output: model - compiled keras model
    '''

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences= False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    #model.summary()
    return model

