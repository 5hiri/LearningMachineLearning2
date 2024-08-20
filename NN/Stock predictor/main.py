import os
import pandas as pd
import seaborn as sns
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, regularization, Conv1D, MaxPooling1D, UpSampling1D, LeakyReLU

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split

os.chdir(os.path.dirname(__file__))

data = pd.read_csv('indexData.csv')

# #Summary
# print(data.describe())

# #Display datatypes
# print(data.dtypes)

# #Remove date column and convert categorical columns
# data = data.drop('Date', axis=1)

#Split the data into their respective index's(stock)
unique_indexs = data['Index'].unique()
# datasets = {}
# for index in unique_indexs:
#     datasets[index] = data[data['Index'] == index]

#     datasets[index].to_csv(f'{index}_dataset.csv', index=False)

def getDataset(index):
    dataset = pd.read_csv(f'{index}_dataset.csv')
    dataset = dataset.dropna() #Drop any null values
    #drop the categorical index value(cause its all the same)
    dataset = dataset.drop('Index', axis=1)
    #create shifted columns
    for i in range(1, 51):
        dataset[f'open_prev_{i}'] = dataset['Open'].shift(i)
        dataset[f'high_prev_{i}'] = dataset['High'].shift(i)
        dataset[f'low_prev_{i}'] = dataset['Low'].shift(i)
        dataset[f'close_prev_{i}'] = dataset['Close'].shift(i)
        dataset[f'adj_close_prev_{i}'] = dataset['Adj Close'].shift(i)
        dataset[f'volume_prev_{i}'] = dataset['Volume'].shift(i)
    for i in range(1, 6):
        dataset[f'open_next_{i}'] = dataset['Open'].shift(-i)
        dataset[f'high_next_{i}'] = dataset['High'].shift(-i)
        dataset[f'low_next_{i}'] = dataset['Low'].shift(-i)
        dataset[f'close_next_{i}'] = dataset['Close'].shift(-i)
        dataset[f'adj_close_next_{i}'] = dataset['Adj Close'].shift(-i)
        dataset[f'volume_next_{i}'] = dataset['Volume'].shift(-i)
    #because of the front and ends of the dataset will have missing next and prev values remove them
    dataset = dataset.copy()
    dataset = dataset.dropna()
    return dataset

# data = getDataset('N100')
# for i in unique_indexs:
#     if i != 'N100':
#         newData = getDataset(i)
#         data = pd.concat([data, newData], axis=0, ignore_index=True)
# data.to_csv('training_data.csv')

data = pd.read_csv('training_data.csv')

#Feature selection
X = data
y_list = []
for i in range(1, 6):
    X = X.drop(f'open_next_{i}', axis=1)
    X = X.drop(f'high_next_{i}', axis=1)
    X = X.drop(f'low_next_{i}', axis=1)
    X = X.drop(f'close_next_{i}', axis=1)
    X = X.drop(f'adj_close_next_{i}', axis=1)
    X = X.drop(f'volume_next_{i}', axis=1)
    y_list.append(f'volume_next_{i}')
    y_list.append(f'adj_close_next_{i}')
    y_list.append(f'close_next_{i}')
    y_list.append(f'low_next_{i}')
    y_list.append(f'high_next_{i}')
    y_list.append(f'open_next_{i}')

y = data[y_list]#.values

#scale x data
# scaler = StandardScaler()
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)#.values

#split test and training data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#Define Model
#Activation Functions:
#elu: Acc = 0.1458190679550171 <-- with msle
#relu: Acc = 0.14385531842708588
#selu: acc = 0.16043293476104736 <-- with msle
#leaky_relu: acc = 0.1402018517255783
model = Sequential([
    Flatten(input_shape=(307, 1)),
    Dense(256, activation='elu'),
    Dense(128, activation='elu'),
    Dense(64, activation='elu'),
    Dense(30, activation='linear')
])

#Compile the model
model.compile(optimizer='adam', loss='msle', metrics=['accuracy'])

#Train the model
history = model.fit(X_train, y_train, batch_size=64, epochs=50, validation_data=(X_test, y_test))

#Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest Accuracy: {test_acc}')

# Predict the labels for test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)