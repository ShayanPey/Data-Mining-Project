#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#CUDA_VISIBLE_DEVICES=""

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


import matplotlib.pyplot as plt
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.layers import Dense, LSTM
import keras
import pydot
import graphviz

#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)
#set_session(sess)

import math
import copy


# In[2]:


if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")


# In[7]:


tf.config.list_physical_devices('CPU')


# In[6]:


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


# In[6]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[3]:


#File='/content/gdrive/My Drive/ML Project/BTC-USD.csv'
File='Desktop/BTC-USD.csv'
df = pd.read_csv(File)
df.info()

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date',inplace=True)

print("SHAPE:", df.shape) #1692 rows and 7 columns that the data frame have 

# create a new data frame with only 'Close column'
data = df.filter(['Close'])
dataset = data.values #convert the data frame to a numpy array

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
#plt.plot(scaled_data,df['Date'],linewidth=3.5)

#plot the data
train = data[:]

#valid = data[training_data_len:training_data_len+29]
#Visialization the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price' ,fontsize=18)
plt.plot(train['Close'],linewidth=3.5)
plt.legend(['Train'], loc='upper left')


# In[ ]:


#File='/content/gdrive/My Drive/ML Project/BTC-USD.csv'
df = pd.read_csv(File)
df.info()

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date',inplace=True)

print("SHAPE:", df.shape) #1692 rows and 7 columns that the data frame have 

# create a new data frame with only 'Close column'
data = df.filter(['Close'])
data=data[-300:]
dataset = data.values #convert the data frame to a numpy array

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
#plt.plot(scaled_data,df['Date'],linewidth=3.5)

#plot the data
train = data[:]

#valid = data[training_data_len:training_data_len+29]
#Visialization the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price' ,fontsize=18)
plt.plot(train['Close'],linewidth=3.5)
plt.legend(['Train'], loc='upper left')


# Making a model for each day in the future

# In[6]:


allPredictions = []
models = [0 for i in range(40)]
for DAY in range(1, 30):
    #For the DAY-th day into the future
    print("~~~~~~~~~~~~~~DAY:"+str(DAY)+ "~~~~~~~~~~~~~~")
    File='/content/gdrive/My Drive/ML Project/BTC-USD.csv'
    File='Desktop/BTC-USD.csv'
    df = pd.read_csv(File)
    df.info()

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date',inplace=True)

    #df = df[767:]#Delete before 2017
    print("SHAPE:", df.shape) #1692 rows and 7 columns that the data frame have 

    # create a new data frame with only 'Close column'
    data = df.filter(['Close'])
    dataset = data.values #convert the data frame to a numpy array
    print("dataset:", dataset[:10])
    #training_data_len = math.ceil(len(dataset)*.8)  # number of rows to train the model on
    #dataRow = 2692 #Last row of data to use to train
    dataRow = df.shape[0]-30
    dataRow = df.shape[0]
    mem = 100 #Short term memory
    k = DAY    #K interval
    training_data_len = dataRow
    training_data_len

    #scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    scaled_data



    XTrain = []
    YTrain = []
    #create the training dataset
    #create the scaled training dataset
    for j in range(k):
        train_data = scaled_data[j:training_data_len:k, :]
        #Split the data into x_train, y_train datasets
        x_train = []
        y_train = []
        for i in range(mem,len(train_data)):
            x_train.append(train_data[i-mem:i, 0])
            y_train.append(train_data[i,0])
        XTrain.append(copy.deepcopy(x_train))
        YTrain.append(copy.deepcopy(y_train))

    x_train = []
    y_train = []
    
    print("len(XTrain[0])", len(XTrain[0]))
    for i in range(len(XTrain[0])): #So that it would be like shifting the series by one
        #Otherwise, the first it would learn one sequence completely, then another
        #This way, it is trained regarding the actual time that it happened
        for j in range(len(XTrain)): #Which must be exactly k
            if len(XTrain[j])>i:
                x_train.append(XTrain[j][i])
                y_train.append(YTrain[j][i])


    #convert the x_train and y_train  to numppy array
    x_train,y_train = np.array(x_train), np.array(y_train)

    #reshape the data
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_train.shape
    
    #print("x_train.shape[1]:", x_train.shape[1])
    #continue

    print("x_train.shape[1]:", x_train.shape[1])
    #Buil the LSTM model
    models[DAY]=Sequential()
    models[DAY].add(LSTM(64,return_sequences=True, input_shape=(x_train.shape[1],1)))
    models[DAY].add(LSTM(64, return_sequences= False))
    models[DAY].add(Dense(32))
    models[DAY].add(Dense(1))
    plt.imshow(keras.utils.plot_model(models[DAY], "my_first_model.png", show_shapes=True))
    break

    #Complie the model
    models[DAY].compile(optimizer='adam', loss='mean_squared_error')

    #Train the model
    models[DAY].fit(x_train,y_train, batch_size=40, epochs=50+int(1.3*(DAY)))

    #get the quote
    quote = pd.read_csv(File)
    #Create new data frame
    new_df = quote.filter(['Close'])
    #get the last mem days closing price values and convert the dataframe to an array
    #last_mem_days = new_df[-mem:].values
    last_mem_days = new_df[dataRow-mem*k:dataRow:k].values

    #scaled the data to be values between 0 and 1
    last_mem_days_scaled = scaler.transform(last_mem_days)
    #create an empty list
    X_test = []
    #append the past mem days 
    X_test.append(last_mem_days_scaled)
    #convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
    #get the predicted scaled price
    pred_price= models[DAY].predict(X_test)
    #undo the scalling
    pred_price = scaler.inverse_transform(pred_price)
    print(pred_price[0][0])
    allPredictions.append(pred_price[0][0])
allPredictions


# In[ ]:


allPredictions = [27466.527,
 26962.186,
 26973.271,
 27703.549,
 28605.902,
 30509.488,
 28938.154,
 29842.365,
 29286.768,
 30639.506,
 30570.924,
 30083.664,
 30648.33,
 31992.654,
 32510.273,
 30811.143,
 29939.637,
 30104.21,
 31189.186,
 30648.14,
 31229.666,
 28901.125,
 30373.174,
 31275.646,
 30772.371,
 31923.164,
 30142.6,
 31974.758,
 29610.12]


# In[ ]:


cop = copy.deepcopy(allPredictions)


# In[ ]:


#plot the data
train = data[training_data_len-300:training_data_len]
print(data[training_data_len-300:training_data_len])
valid = data[training_data_len:training_data_len+29]
valid['Predictions'] = allPredictions
#Visialization the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price' ,fontsize=18)
plt.plot(train['Close'],linewidth=3.5)
plt.plot(valid[['Predictions', 'Close']],linewidth=3.5)
#plt.legend(['Train','Valid','Predictions'], loc='upper left')
plt.legend(['Train','Predictions', 'Close'], loc='upper left')


# In[ ]:





# In[ ]:





# In[ ]:


#Predicing the past data
#Predicing the data which the model trained on.

pastPredictions = []
endRow = dataRow-50
for k in range(1,30):
    #get the quote
    quote = pd.read_csv(File)
    #Create new data frame
    new_df = quote.filter(['Close'])
    #get the last mem days closing price values and convert the dataframe to an array
    #last_mem_days = new_df[-mem:].values
    last_mem_days = new_df[endRow-mem*k:endRow:k].values

    #scaled the data to be values between 0 and 1
    last_mem_days_scaled = scaler.transform(last_mem_days)


    #last_mem_days_scaled = scaled_data[endRow-mem*k:endRow:k]
    #create an empty list
    X_test = []
    #append the past mem days 
    X_test.append(last_mem_days_scaled)
    #convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
    #get the predicted scaled price
    pred_price= models[k].predict(X_test)
    #undo the scalling
    pred_price = scaler.inverse_transform(pred_price)
    print(pred_price[0][0])
    pastPredictions.append(pred_price[0][0])
pastPredictions


# In[ ]:


#Past Predictions
train = data[training_data_len-300:]
valid = data[endRow:endRow+29]
valid['Predictions'] = pastPredictions
#Visialization the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price' ,fontsize=18)
plt.plot(train['Close'],linewidth=3.5)
plt.plot(valid[['Predictions', 'Close']],linewidth=3.5)
plt.legend(['Train','Predictions', 'Close'], loc='upper left')


# In[ ]:


#Foreseeing the future
endRow = df.shape[0]


File='/content/gdrive/My Drive/ML Project/BTC-USD-Future.csv' #Contains dates for the future
df = pd.read_csv(File)
df.info()

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date',inplace=True)

print("SHAPE:", df.shape)

# create a new data frame with only 'Close column'
data = df.filter(['Close'])
futurePredictions = []
print("df.shape[0]:", df.shape[0])
#endRow = df.shape[0]
for k in range(1,30):
    #get the quote
    quote = pd.read_csv(File)
    #Create new data frame
    new_df = quote.filter(['Close'])
    #get the last mem days closing price values and convert the dataframe to an array
    #last_mem_days = new_df[-mem:].values
    last_mem_days = new_df[endRow-mem*k:endRow:k].values

    #scaled the data to be values between 0 and 1
    last_mem_days_scaled = scaler.transform(last_mem_days)


    #last_mem_days_scaled = scaled_data[endRow-mem*k:endRow:k]
    #create an empty list
    X_test = []
    #append the past mem days 
    X_test.append(last_mem_days_scaled)
    #convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
    #get the predicted scaled price
    pred_price= models[k].predict(X_test)
    #undo the scalling
    pred_price = scaler.inverse_transform(pred_price)
    print(pred_price[0][0])
    futurePredictions.append(pred_price[0][0])
futurePredictions


# In[8]:


#Future Predictions PLOT

futurePredictions = allPredictions #SHOULD BE REMOVED. Just for when bypassed the test data and straight to predicting the future
endRow = dataRow #SHOULD BE REMOVED
File='/content/gdrive/My Drive/ML Project/BTC-USD-Future.csv' #Contains dates for the future
df = pd.read_csv(File)
df.info()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date',inplace=True)
# create a new data frame with only 'Close column'
data = df.filter(['Close'])



train = data[-60:]
valid = data[endRow:endRow+29]
valid['Predictions'] = futurePredictions
#Visialization the data
plt.figure(figsize=(16,8))
plt.grid()
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price' ,fontsize=18)
plt.plot(train['Close'],linewidth=3.5)
plt.plot(valid[['Predictions', 'Close']],linewidth=3.5)
plt.legend(['Train','Predictions', 'Close'], loc='upper left')


# In[ ]:





# **Mean of the interval**

# In[ ]:


allPredictions = []
models = [0 for i in range(40)]
for DAY in range(1, 30):
    #For the DAY-th day into the future
    print("~~~~~~~~~~~~~~DAY:"+str(DAY)+ "~~~~~~~~~~~~~~")
    File='/content/gdrive/My Drive/ML Project/BTC-USD.csv'
    df = pd.read_csv(File)
    df.info()

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date',inplace=True)

    #df = df[767:]#Delete before 2017
    print("SHAPE:", df.shape) #1692 rows and 7 columns that the data frame have 

    # create a new data frame with only 'Close column'
    data = df.filter(['Close'])
    dataset = data.values #convert the data frame to a numpy array
    print("dataset:", dataset[:10])
    #training_data_len = math.ceil(len(dataset)*.8)  # number of rows to train the model on
    #dataRow = 2692 #Last row of data to use to train
    dataRow = df.shape[0]-30
    mem = 100 #Short term memory
    k = DAY    #K interval
    training_data_len = dataRow
    training_data_len

    #scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    scaled_data

    x_train = []
    y_train = []
    meanData = []
    for i in range(k, training_data_len, +k):
        meanData.append(np.mean(np.array(scaled_data[i-k:i, 0])))
    for i in range(mem, len(meanData)-1):
      x_train.append(copy.deepcopy(meanData[i-mem:i]))
      y_train.append(meanData[i+1])

    #convert the x_train and y_train  to numppy array
    x_train,y_train = np.array(x_train), np.array(y_train)

    #reshape the data
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_train.shape
    
    #print("x_train.shape[1]:", x_train.shape[1])

    print("x_train.shape[1]:", x_train.shape[1])
    #Buil the LSTM model
    models[DAY]=Sequential()
    models[DAY].add(LSTM(64,return_sequences=True, input_shape=(x_train.shape[1],1)))
    models[DAY].add(LSTM(64, return_sequences= False))
    models[DAY].add(Dense(32))
    models[DAY].add(Dense(1))

    #Complie the model
    models[DAY].compile(optimizer='adam', loss='mean_squared_error')

    #Train the model
    models[DAY].fit(x_train,y_train, batch_size=40, epochs=30)

    #get the quote
    quote = pd.read_csv(File)
    #Create new data frame
    new_df = quote.filter(['Close'])
    #get the last mem days closing price values and convert the dataframe to an array
    #last_mem_days = new_df[-mem:].values
    last_mem_days = meanData[-mem:]
    last_mem_days_scaled = last_mem_days

    #scaled the data to be values between 0 and 1
    #last_mem_days_scaled = scaler.transform(last_mem_days)
    #create an empty list
    X_test = []
    #append the past mem days 
    X_test.append(last_mem_days_scaled)
    #convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
    #get the predicted scaled price
    pred_price= models[DAY].predict(X_test)
    #undo the scalling
    pred_price = scaler.inverse_transform(pred_price)
    print(pred_price[0][0])
    allPredictions.append(pred_price[0][0])
allPredictions


# In[ ]:


#plot the data for the last 30 days
train = data[training_data_len-300:training_data_len]
print(data[training_data_len-300:training_data_len])
valid = data[training_data_len:training_data_len+29]
valid['Predictions'] = allPredictions
#Visialization the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price' ,fontsize=18)
plt.plot(train['Close'],linewidth=3.5)
plt.plot(valid[['Predictions', 'Close']],linewidth=3.5)
#plt.legend(['Train','Valid','Predictions'], loc='upper left')
plt.legend(['Train','Predictions', 'Close'], loc='upper left')


# In[ ]:


#For the DAY-th day into the future
File='/content/gdrive/My Drive/ML Project/BTC-USD.csv'
df = pd.read_csv(File)
df.info()

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date',inplace=True)

#df = df[767:]#Delete before 2017
print("SHAPE:", df.shape) #1692 rows and 7 columns that the data frame have 

# create a new data frame with only 'Close column'
data = df.filter(['Close'])
dataset = data.values #convert the data frame to a numpy array
print("dataset:", dataset[:10])
#training_data_len = math.ceil(len(dataset)*.8)  # number of rows to train the model on
#dataRow = 2692 #Last row of data to use to train
dataRow = df.shape[0]-30
mem = 100 #Short term memory
k = 40    #K interval
training_data_len = dataRow
training_data_len

#scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data

x_train = []
y_train = []
meanData = []
for i in range(k, training_data_len, +k):
    meanData.append(np.mean(np.array(scaled_data[i-k:i, 0])))


#convert the x_train and y_train  to numppy array
x_train,y_train = np.array(x_train), np.array(y_train)

plt.figure(figsize=(16,8))
#plt.plot([i for i in range(len(dataset))], dataset)
plt.plot(meanData)


# In[ ]:





# Predicing based on previous predictions

# In[ ]:


allPredictions = []
models = [0, 0, 0, 0]

DAY = 1
#For the DAY-th day into the future
print("~~~~~~~~~~~~~~DAY:"+str(DAY)+ "~~~~~~~~~~~~~~")
File='/content/gdrive/My Drive/ML Project/BTC-USD.csv'
df = pd.read_csv(File)
df.info()

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date',inplace=True)

#df = df[767:]#Delete before 2017
print("SHAPE:", df.shape) #1692 rows and 7 columns that the data frame have 

# create a new data frame with only 'Close column'
data = df.filter(['Close'])
dataset = data.values #convert the data frame to a numpy array
print("dataset:", dataset[:10])
#training_data_len = math.ceil(len(dataset)*.8)  # number of rows to train the model on
#dataRow = 2692 #Last row of data to use to train
dataRow = df.shape[0]-30
mem = 100 #Short term memory
k = DAY    #K interval
training_data_len = dataRow
training_data_len

#scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data



XTrain = []
YTrain = []
#create the training dataset
#create the scaled training dataset
for j in range(k):
    train_data = scaled_data[j:training_data_len:k, :]
    #Split the data into x_train, y_train datasets
    x_train = []
    y_train = []
    for i in range(mem,len(train_data)):
        x_train.append(train_data[i-mem:i, 0])
        y_train.append(train_data[i,0])
    XTrain.append(copy.deepcopy(x_train))
    YTrain.append(copy.deepcopy(y_train))

x_train = []
y_train = []

print("len(XTrain[0])", len(XTrain[0]))
for i in range(len(XTrain[0])): #So that it would be like shifting the series by one
    #Otherwise, the first it would learn one sequence completely, then another
    #This way, it is trained regarding the actual time that it happened
    for j in range(len(XTrain)): #Which must be exactly k
        if len(XTrain[j])>i:
            x_train.append(XTrain[j][i])
            y_train.append(YTrain[j][i])


#convert the x_train and y_train  to numppy array
x_train,y_train = np.array(x_train), np.array(y_train)

#reshape the data
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape

#print("x_train.shape[1]:", x_train.shape[1])

print("x_train.shape[1]:", x_train.shape[1])
#Buil the LSTM model
models[DAY]=Sequential()
models[DAY].add(LSTM(64,return_sequences=True, input_shape=(x_train.shape[1],1)))
models[DAY].add(LSTM(64, return_sequences= False))
models[DAY].add(Dense(32))
models[DAY].add(Dense(1))

#Complie the model
models[DAY].compile(optimizer='adam', loss='mean_squared_error')

#Train the model
models[DAY].fit(x_train,y_train, batch_size=50, epochs=20)

#get the quote
quote = pd.read_csv(File)
#Create new data frame
new_df = quote.filter(['Close'])
#get the last mem days closing price values and convert the dataframe to an array
#last_mem_days = new_df[-mem:].values
last_mem_days = new_df[dataRow-mem*k:dataRow:k].values

#scaled the data to be values between 0 and 1
last_mem_days_scaled = scaler.transform(last_mem_days)
#create an empty list
X_test = []
#append the past mem days 
X_test.append(last_mem_days_scaled)
#convert the X_test data set to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
#get the predicted scaled price
pred_price= models[DAY].predict(X_test)
#undo the scalling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price[0][0])
allPredictions.append(pred_price[0][0])
allPredictions


# In[ ]:


#Predicting the next 30 days based on previous predictions
quote = pd.read_csv(File)
#Create new data frame
new_df = quote.filter(['Close'])
endRow = new_df.shape[0]-30
pastPredictions = new_df[endRow-mem:endRow].values
pastPredictions = scaler.transform(pastPredictions)

forPlot = []

for k in range(1,30):
    #last_mem_days_scaled = scaled_data[endRow-mem*k:endRow:k]
    #create an empty list
    X_test = []
    #append the past mem days 
    X_test.append(pastPredictions[k:])
    #convert the X_test data set to a numpy array
    X_test = np.array(X_test)
    #Reshape the data
    X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
    #get the predicted scaled price
    pred_price= models[1].predict(X_test)
    #undo the scalling
    pastPredictions=np.append(pastPredictions, pred_price[0][0])
    print(len(pastPredictions))
    pred_price = scaler.inverse_transform(pred_price)
    print(pred_price[0][0])
    forPlot.append(pred_price[0][0])

pastPredictions = pastPredictions[-29:]
pastPredictions = scaler.inverse_transform(pastPredictions.reshape(-1,1))
pastPredictions


# In[ ]:


#Last Predictions
train = data[training_data_len-300:]
valid = data[endRow:endRow+29]
valid['Predictions'] = pastPredictions
#Visialization the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price' ,fontsize=18)
plt.plot(train['Close'],linewidth=3.5)
plt.plot(valid[['Predictions', 'Close']],linewidth=3.5)
plt.legend(['Train','Predictions', 'Close'], loc='upper left')


# In[ ]:





# Approach 3: Using a model with 30 neurons in the output layer, which the ith layer is responsible for predicting the price of bitcon for the ith day in the future.

# In[ ]:


allPredictions = []
models = [0 for i in range(40)]

#For the DAY-th day into the future
print("~~~~~~~~~~~~~~DAY:"+str(DAY)+ "~~~~~~~~~~~~~~")
File='/content/gdrive/My Drive/ML Project/BTC-USD.csv'
df = pd.read_csv(File)
df.info()

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date',inplace=True)

#df = df[767:]#Delete before 2017
print("SHAPE:", df.shape) #1692 rows and 7 columns that the data frame have 

# create a new data frame with only 'Close column'
data = df.filter(['Close'])
dataset = data.values #convert the data frame to a numpy array
print("dataset:", dataset[:10])
#training_data_len = math.ceil(len(dataset)*.8)  # number of rows to train the model on
#dataRow = 2692 #Last row of data to use to train
dataRow = df.shape[0]-30
mem = 100 #Short term memory
k = DAY    #K interval
training_data_len = dataRow
training_data_len
#scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data



XTrain = []
YTrain = []
#create the training dataset
#create the scaled training dataset
for j in range(k):
    train_data = scaled_data[j:training_data_len:k, :]
    #Split the data into x_train, y_train datasets
    x_train = []
    y_train = []
    for i in range(mem,len(train_data)-30):
        x_train.append(train_data[i-mem:i, 0])
        y_train.append(train_data[i:i+30,0])
    XTrain.append(copy.deepcopy(x_train))
    YTrain.append(copy.deepcopy(y_train))

x_train = []
y_train = []


#convert the x_train and y_train  to numppy array
x_train,y_train = np.array(x_train), np.array(y_train)

#reshape the data
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape

#print("x_train.shape[1]:", x_train.shape[1])

print("x_train.shape[1]:", x_train.shape[1])
#Buil the LSTM model
models[DAY]=Sequential()
models[DAY].add(LSTM(64,return_sequences=True, input_shape=(x_train.shape[1],1)))
models[DAY].add(LSTM(64, return_sequences= False))
models[DAY].add(Dense(32))
models[DAY].add(Dense(30))

#Complie the model
models[DAY].compile(optimizer='adam', loss='mean_squared_error')

#Train the model
models[DAY].fit(x_train,y_train, batch_size=40, epochs=30)

#get the quote
quote = pd.read_csv(File)
#Create new data frame
new_df = quote.filter(['Close'])
#get the last mem days closing price values and convert the dataframe to an array
#last_mem_days = new_df[-mem:].values
last_mem_days = new_df[dataRow-mem*k:dataRow:k].values

#scaled the data to be values between 0 and 1
last_mem_days_scaled = scaler.transform(last_mem_days)
#create an empty list
X_test = []
#append the past mem days 
X_test.append(last_mem_days_scaled)
#convert the X_test data set to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
#get the predicted scaled price
pred_price= models[DAY].predict(X_test)
#undo the scalling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price[0][0])
allPredictions.append(pred_price[0][0])
allPredictions


# In[ ]:





# In[ ]:





# In[ ]:


def cost1(pred, close):
    sum = 0
    for i in range(len(pred)):
        sum+=abs(pred[i]-close[i])
    return sum/30

def cost2(pred, close):
    sum = 0
    for i in range(len(pred)):
        sum+= (pred[i]-close[i])**2
    return sum/30

def cost3(pred, close):
    sum = 0
    for i in range(len(pred)):
        sum += (30-i)*abs(pred[i]-close[i])
    return sum/30


# In[ ]:


#Costs
#pastPredictions = allPredictions

quote = pd.read_csv(File)
#Create new data frame
new_df = quote.filter(['Close'])
endRow = new_df.shape[0]-30

valid = data[endRow:endRow+29]
close = valid['Close']

print("cost 1:", cost1(pastPredictions, close), "cost 2:", cost2(pastPredictions, close),"cost 3:", cost3(pastPredictions, close),)


# In[ ]:


#Last Predictions
train = data[training_data_len-300:]
valid = data[endRow:endRow+29]
valid['Predictions'] = pastPredictions
#Visialization the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price' ,fontsize=18)
plt.plot(train['Close'],linewidth=3.5)
plt.plot(valid[['Predictions', 'Close']],linewidth=3.5)
plt.legend(['Train','Predictions', 'Close'], loc='upper left')


# In[ ]:




