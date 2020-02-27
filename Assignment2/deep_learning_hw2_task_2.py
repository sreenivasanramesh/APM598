from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

# Design of Neural Network

# Neural Network for 2 data-points
# MLPRegressor uses ReLu activation by default
def get_model_task1():
    neural_network = MLPRegressor(hidden_layer_sizes = (1024, 512), solver = 'sgd', learning_rate_init = 0.001, batch_size = 5, max_iter = 500, verbose = True)
    return neural_network

# Neural Network for 3 data-points
def get_model_task2():
    neural_network = MLPRegressor(hidden_layer_sizes = (1024, 512), solver = 'sgd', learning_rate_init = 0.001, batch_size = 5, max_iter = 500, verbose = True)
    return neural_network

# Neural Network for n data-points
def get_model_task3():
    neural_network = MLPRegressor(hidden_layer_sizes =( 256, 128), solver='sgd', learning_rate_init = 0.001, batch_size = 5, max_iter = 500, verbose = True)
    return neural_network


# This function generates continuous values
# Define the function that you want as "input"
# This is g(x)
def generate_continuous_values(x):
    #define the function that you want as "input"
    return 1/(1 + np.exp(-x))

def call_to_g_of_x(input, g_of_x):
    result = g_of_x(input)
    return result


# Part 1 - Two data points
def get_two_data_points(point_1, point_2):
    x = [point_1, point_2]
    x = np.asarray(x)
    x.shape = (len(x),1)
    y = np.array([call_to_g_of_x(x[i], generate_continuous_values) for i in range(len(x))])
    y.shape = (len(y),1)
    print(x,y)
    return x,y


# Part 2 - Three data points
def get_three_data_points(point_1, point_2, point_3):
    x = [point_1, point_2, point_3]
    x = np.asarray(x)
    x.shape = (len(x),1)
    y = np.array([call_to_g_of_x(x[i], generate_continuous_values) for i in range(len(x))])
    y.shape = (len(y),1)
    print(x,y)
    return x,y


# Part 3 - n data points
def get_n_data_points(start, end, step_size):
    x = np.arange(start, end, step_size)
    x.shape = (len(x), 1)
    y = np.array([call_to_g_of_x(x[i], generate_continuous_values) for i in range(len(x))])
    y.shape = (len(y), 1)
    print(x,y)
    return x,y




# Train Model
x_1, y_1 = get_two_data_points(0, 1)
x_2, y_2 = get_three_data_points(0, 0.5, 1) #function call for part 2 data prep
x_3, y_3 = get_n_data_points(0, 1, .01) #function call for part 3 data prep


# Biuld and fit model for part 1
neural_network = get_model_task3()
neural_network.fit(x_1, y_1)

print("TASK 1 MODEL SUMMARY AND WEIGHTS---")
print("Weights:")
print(neural_network.coefs_) #print WEIGHTS

print("Number of layers:")
print(neural_network.n_layers_) #prints number of layers

print("Bias:")
print(neural_network.intercepts_) #prints the BIAS

weights = neural_network.coefs_
for i in range(len(weights)):
    print("Layer " + str(i) + ":" + str(len(weights[i])) + " nodes")

predictions = neural_network.predict(x_1)

plt.title("Continuous Function for 2 data points")
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.plot(x_1, y_1, label = 'g(x)')
plt.plot(x_1, predictions, label = 'f(x)')
plt.legend()
plt.savefig("task1.png")
plt.clf()



# Build and fit model for part 2
neural_network = get_model_task3()
neural_network.fit(x_2, y_2)

print("TASK 2 MODEL SUMMARY AND WEIGHTS---")
print("Weights:")
print(neural_network.coefs_)

# Total 4 layes : input, 2 hidden layers and one output layer
print("Number of layers:")
print(neural_network.n_layers_) #prints number of layers

print("Bias:")
print(neural_network.intercepts_)

weights = neural_network.coefs_
for i in range(len(weights)):
    print("Layer " + str(i) + ":" + str(len(weights[i])) + " nodes")

predictions = neural_network.predict(x_2)

plt.title("Continuous Function for 3 data points")
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.plot(x_2, y_2, label = 'g(x)')
plt.plot(x_2, predictions, label = 'f(x)')
plt.legend()
plt.savefig("task2.png")
plt.clf()


# Build and fit model for Part 3
neural_network = get_model_task3()
neural_network.fit(x_3, y_3)

print("TASK 3 MODEL SUMMARY AND WEIGHTS---")
print("Weights:")
print(neural_network.coefs_) #print WEIGHTS

print("Number of layers:")
print(neural_network.n_layers_) #prints number of layers

print("Bias:")
print(neural_network.intercepts_) #prints the BIAS

weights = neural_network.coefs_
for i in range(len(weights)):
    print("Layer " + str(i) + ":" + str(len(weights[i])) + " nodes")

predictions = neural_network.predict(x_3)

plt.title("Continuous Function for n data points")
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.plot(x_3, y_3, label = 'g(x)')
plt.plot(x_3, predictions, label = 'f(x)')
plt.legend()
plt.savefig("task3.png")
plt.clf()
