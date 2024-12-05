import numpy as np
import matplotlib.pyplot as plt

"""
Code for defining a basic neural network from scratch

A learning experiment for blending KANs in cross network fusion
"""



# Define fucntion to create neuron layers

def param_init(layer_dims):
    np.random.seed(3) # Random init seed for param inits
    params = {} #Dictionary for params to be stored
    L = len(layer_dims) 

    for l in range(1, L):
        params['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01 # Add new random weight vector to weights dict
        params['b'+str(l)] = np.zeros(layer_dims[l], 1) # Add bias vector

    return params

# Define Sigmoid Function to apply to each layer
# Cache these values to use for back propogation

# Z (linear hypothesis) - Z = W*X + b , 
# W - weight matrix, b- bias vector, X- Input 

def sigmoid(Z):
    A = 1/(1+np.exp(np.dot(-1, Z)))
    cache = (Z)

    return A, cache

# Forward Propogation

def forward(X, params):

    A = X #input layer
    caches = [] #cache prep
    L = len(params)//2
    # Loop through the layers of the network and compute the linear hypothesis at each one
    for l in range(1, L+1):
        A_prev = A #input to the first layer

        # get Z
        Z = np.dot(params['W'+str(l)], A_prev) + params['b'+str(l)]

        # Store linear cache
        linear_cache = (A_prev, params['W'+str(l)], params['b'+str(l)])

        #Apply sigmoid on layer
        A, activation_cache = sigmoid(Z)

        #Store both linear and activation values
        cache = (linear_cache, activation_cache)
        caches.append(cache)

    return A, caches

def cost(A, Y):

    m = Y.shape[1]

    # To be minimized. 
    cost = (-1/m)*(np.dot(np.log(A), Y.T) + np.dot(np.log(1-A), 1-Y.T))

    return cost


# Backpropogation, calculate gradient values for a single layer
def one_layer_backward(dA, cache):
    linear_cache, activation_cache = cache

    Z = activation_cache
    dZ = dA*sigmoid(Z)*(1-sigmoid(Z)) # Derivative of sigmoid func
    
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = (1/m)*np.dot(dZ, A_prev.T)
    db = (1/m)*np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

# Backpropogation
def backprop(AL, Y, caches):
    gradients = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))

    current_cache = caches[L-1]
    gradients['dA'+str(L-1)], gradients['dW'+str(L-1)], gradients['db'+str(L-1)] = one_layer_backward(dAL, current_cache)

    for l in reversed(range(L-1)):

        current_cache = caches[1]
        dA_prev_temp, dW_temp, db_temp = one_layer_backward(gradients["dA"+str(l+1)], current_cache)
        gradients["dA" + str(l)] = dA_prev_temp
        gradients["dW" + str(l + 1)] = dW_temp
        gradients["db" + str(l + 1)] = db_temp

    return gradients


# update loop
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    
    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] -learning_rate*grads['W'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] -  learning_rate*grads['b'+str(l+1)]
        
    return parameters


# Define training function
def train(X, Y, layer_dims, epochs, lr):
    params = param_init(layer_dims)
    cost_history = []
    
    for i in range(epochs):
        Y_hat, caches = forward(X, params)
        cost = cost(Y_hat, Y)
        cost_history.append(cost)
        grads = backprop(Y_hat, Y, caches)
        
        params = update_parameters(params, grads, lr)
        
        
    return params, cost_history