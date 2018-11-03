
# Part 1 - Data Preprocessing


```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```


```python
# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values
```


```python
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
```


```python
# Creating a data structure with 60 timesteps and t+1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
```


```python
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
```

# Part 2 - Building the RNN


```python
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
```

    C:\Users\Hemanth\Anaconda3\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.
    


```python
# Initialising the RNN
regressor = Sequential()
```


```python
# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 3, return_sequences = True, input_shape = (None, 1)))
```


```python
# Adding a second LSTM layer
regressor.add(LSTM(units = 3, return_sequences = True))
```


```python
# Adding a third LSTM layer
regressor.add(LSTM(units = 3, return_sequences = True))
```


```python
# Adding a fourth LSTM layer
regressor.add(LSTM(units = 3))
```


```python
# Adding the output layer
regressor.add(Dense(units = 1))
```


```python
# Compiling the RNN
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')
```


```python
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
```

    Epoch 1/100
    1198/1198 [==============================] - 8s 7ms/step - loss: 0.1551
    Epoch 2/100
    1198/1198 [==============================] - 5s 4ms/step - loss: 0.0283
    Epoch 3/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0056
    Epoch 4/100
    1198/1198 [==============================] - 5s 4ms/step - loss: 0.0041
    Epoch 5/100
    1198/1198 [==============================] - 5s 4ms/step - loss: 0.0034
    Epoch 6/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0030
    Epoch 7/100
    1198/1198 [==============================] - 5s 4ms/step - loss: 0.0029
    Epoch 8/100
    1198/1198 [==============================] - 5s 4ms/step - loss: 0.0027
    Epoch 9/100
    1198/1198 [==============================] - 5s 4ms/step - loss: 0.0026
    Epoch 10/100
    1198/1198 [==============================] - 5s 4ms/step - loss: 0.0026
    Epoch 11/100
    1198/1198 [==============================] - 5s 4ms/step - loss: 0.0026
    Epoch 12/100
    1198/1198 [==============================] - 5s 4ms/step - loss: 0.0025
    Epoch 13/100
    1198/1198 [==============================] - 5s 4ms/step - loss: 0.0024
    Epoch 14/100
    1198/1198 [==============================] - 5s 4ms/step - loss: 0.0024
    Epoch 15/100
    1198/1198 [==============================] - 5s 4ms/step - loss: 0.0022
    Epoch 16/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0021
    Epoch 17/100
    1198/1198 [==============================] - 5s 4ms/step - loss: 0.0022
    Epoch 18/100
    1198/1198 [==============================] - 5s 4ms/step - loss: 0.0022
    Epoch 19/100
    1198/1198 [==============================] - 5s 4ms/step - loss: 0.0021
    Epoch 20/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0021
    Epoch 21/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0020
    Epoch 22/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0020
    Epoch 23/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0019
    Epoch 24/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0019
    Epoch 25/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0018
    Epoch 26/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0018
    Epoch 27/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0017
    Epoch 28/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0018
    Epoch 29/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0017
    Epoch 30/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0017
    Epoch 31/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0017
    Epoch 32/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0016
    Epoch 33/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0016
    Epoch 34/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0018
    Epoch 35/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0016
    Epoch 36/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0015
    Epoch 37/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0015
    Epoch 38/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0017
    Epoch 39/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0015
    Epoch 40/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0015
    Epoch 41/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0015
    Epoch 42/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0015
    Epoch 43/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0015
    Epoch 44/100
    1198/1198 [==============================] - 5s 4ms/step - loss: 0.0015
    Epoch 45/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0014
    Epoch 46/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0015
    Epoch 47/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0015
    Epoch 48/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0014
    Epoch 49/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0014
    Epoch 50/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0014
    Epoch 51/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0013
    Epoch 52/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0014
    Epoch 53/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0014
    Epoch 54/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0014A: 
    Epoch 55/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0014
    Epoch 56/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0014
    Epoch 57/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0014
    Epoch 58/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0014
    Epoch 59/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0014
    Epoch 60/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0014
    Epoch 61/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0013
    Epoch 62/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0013
    Epoch 63/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0013
    Epoch 64/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0013
    Epoch 65/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0014
    Epoch 66/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0013
    Epoch 67/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0012
    Epoch 68/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0013
    Epoch 69/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0012
    Epoch 70/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0013
    Epoch 71/100
    1198/1198 [==============================] - 5s 4ms/step - loss: 0.0012
    Epoch 72/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0013
    Epoch 73/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0013
    Epoch 74/100
    1198/1198 [==============================] - 5s 4ms/step - loss: 0.0012
    Epoch 75/100
    1198/1198 [==============================] - 5s 4ms/step - loss: 0.0012
    Epoch 76/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0012
    Epoch 77/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0012
    Epoch 78/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0012
    Epoch 79/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0012
    Epoch 80/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0011
    Epoch 81/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0012
    Epoch 82/100
    1198/1198 [==============================] - 5s 4ms/step - loss: 0.0011
    Epoch 83/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0011
    Epoch 84/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0011
    Epoch 85/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0011
    Epoch 86/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0010
    Epoch 87/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0012
    Epoch 88/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0011
    Epoch 89/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0011
    Epoch 90/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0010
    Epoch 91/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0011
    Epoch 92/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0011
    Epoch 93/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 9.9218e-04
    Epoch 94/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0010
    Epoch 95/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0010
    Epoch 96/100
    1198/1198 [==============================] - 4s 4ms/step - loss: 0.0010
    Epoch 97/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 9.4644e-04
    Epoch 98/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 0.0010
    Epoch 99/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 9.9575e-04
    Epoch 100/100
    1198/1198 [==============================] - 4s 3ms/step - loss: 9.2934e-04
    




    <keras.callbacks.History at 0x197aa0c0eb8>




```python
regressor.save('saved_rnn_60timesteps_4lstmlayers.h5')
```

# Part 3 - Making the predictions and visualising the results


```python
# Getting the real stock price for February 1st 2012 - January 31st 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
test_set = dataset_test.iloc[:,1:2].values
real_stock_price = np.concatenate((training_set[0:1258], test_set), axis = 0)
```


```python
# Getting the predicted stock price of 2017
scaled_real_stock_price = sc.fit_transform(real_stock_price)
inputs = []
for i in range(1258, 1278):
    inputs.append(scaled_real_stock_price[i-60:i, 0])
inputs = np.array(inputs)
inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
```


```python
# Visualising the results
plt.plot(real_stock_price[1258:], color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```


![png](output_20_0.png)

