import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
import gc
import tensorflow as tf

######################### Part 1 of Ex 1######################################################



x = np.array([[1,0],[-1,0],[0,1],[0,-1]])
y = np.array([[1000,-1000],[1000,-1000],[-1000,1000],[-1000,1000]])



model = Sequential()
model.add(Dense(4, input_shape = (2,), activation = 'relu'))
model.add(Dense(2))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=None)

model.fit(x, y, epochs=100000)
#Get Weights and biases of layer 1

weights_1 = model.layers[0].get_weights()[0]
biases_1 = model.layers[0].get_weights()[1]

#Get Weights and biases of layer 2

weights_2 = model.layers[1].get_weights()[0]
biases_2 = model.layers[1].get_weights()[1]

print(weights_1)
print(biases_1)
print(weights_2)
print(biases_2)

print(model.predict(x))






'''

########################## Part 2 of Ex 1#######################################################

df = pd.read_csv('data_HW2_ex1.csv')
x = np.column_stack((df['x1'].values,df['x2'].values))
y = df['class'].values
#y = to_categorical(y)


print(x.shape)
print(y.shape)
print(x)
print(y)


#Define the fully connected model

model = Sequential()
model.add(Dense(4, input_shape = (2,), activation = 'relu'))
model.add(Dense(1))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, epochs=1000)

# evaluate the keras model
score, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))


#Get Weights and biases of layer 1

weights_1 = model.layers[0].get_weights()[0]
biases_1 = model.layers[0].get_weights()[1]

#Get Weights and biases of layer 2

weights_2 = model.layers[1].get_weights()[0]
biases_2 = model.layers[1].get_weights()[1]

print(weights_1)
print(biases_1)
print(weights_2)
print(biases_2)
'''

del model
gc.collect()